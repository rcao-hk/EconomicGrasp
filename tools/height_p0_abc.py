#!/usr/bin/env python3
"""P0-A epoch generalization, P0-B height-bias/constant baseline, P0-C paired RGB.

Example:
    python tools/height_p0_abc.py fit-prior --dataset-root ROOT --output prior.json
    python tools/height_p0_abc.py evaluate --dataset-root ROOT \
        --gntrans-rgb-root GNTRANS --checkpoints checkpoints.json \
        --prior prior.json --appearance --output-dir RUN
    python tools/height_p0_abc.py summarize --output-dir RUN

The inference model is ONLY the checkpoint's frozen DPT depth submodule.
No RayRoPE/GVAR action evaluation, no new optimization, no official grasp AP.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

# Ensure root imports work when launched as python tools/height_p0_abc.py.
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from diagnostics.height_p0.checkpoints import (
    read_checkpoint_registry,load_depth_model,predict,sha256,
)
from diagnostics.height_p0.data import build_registry
from diagnostics.height_p0.metrics import (
    SCHEMA,SPLITS,aggregate,frame_metrics,paired_appearance_metrics,
    object_pool,stable_seed,
)


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,path)


def code_hashes():
    paths=[
        "tools/height_p0_abc.py",
        "diagnostics/height_p0/metrics.py",
        "diagnostics/height_p0/data.py",
        "diagnostics/height_p0/checkpoints.py",
        "dataset/graspnet_dataset.py",
        "models/economicgrasp_depth.py",
        "height_analytic_geometry.py",
    ]
    return {p:sha256(ROOT/p) for p in paths}


def ray_dot_from_K(K,plane,shape):
    """Pinhole r_z=1, exactly the height decoder's K^-1 computation."""
    if K.shape!=(3,3) or plane.shape!=(4,):
        raise ValueError("Bad K/plane geometry.")
    if not (np.isfinite(K).all() and np.isfinite(plane).all()):
        raise ValueError("Nonfinite K/plane.")
    H,W=map(int,shape)
    y,x=np.meshgrid(np.arange(H),np.arange(W),indexing="ij")
    homogeneous=np.stack((x,y,np.ones_like(x)),-1).astype(np.float64)
    inv=np.linalg.inv(K.astype(np.float64))
    ray=homogeneous@inv.T
    ray/=ray[...,2:3]
    dot=ray@plane[:3].astype(np.float64)
    return dot.astype(np.float32)


def read_prior(path,root):
    p=json.loads(Path(path).read_text())
    if (p.get("schema")!=SCHEMA or p.get("partition")!="train_used" or
        p.get("gt_source")!="rendered_object_foreground" or
        Path(p.get("dataset_root","")).resolve()!=Path(root).resolve()):
        raise ValueError("Constant-height prior must be fit only on original training frames and same dataset.")
    h0=float(p["h0_m"])
    if not np.isfinite(h0):raise ValueError("Nonfinite train-only prior.")
    return h0,p


def fit_prior(args):
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    if args.frames_per_scene>128:raise ValueError("Max 128 even train frames per scene.")
    ds=build_registry(args.dataset_root,None,("train_used",),args.frames_per_scene)
    values=[];nframes=0;nvalid=0;discard=0
    started=perf_counter()
    for j in range(len(ds)):
        s=ds.for_prior(j)
        z=s["gt"].numpy()
        seg=s["seg"].numpy()
        dot=ray_dot_from_K(s["K"].numpy(),s["plane"].numpy(),z.shape)
        mask=(seg>0)&(z>=.2)&(z<=1.)&(dot<-.1)
        q=(dot*z+float(s["plane"][3]))[mask]
        if not len(q):
            raise ValueError(f"Empty train-reference foreground: {s['scene']} frame {s['frame']}")
        rng=np.random.default_rng(stable_seed(["prior",s["scene"],s["frame"],args.seed]))
        if len(q)>args.pixels_per_frame:
            q=q[rng.choice(len(q),args.pixels_per_frame,replace=False)]
        values.append(q.astype(np.float32,copy=True))
        nvalid+=len(q);nframes+=1
        if nframes%200==0:print(f"[prior] {nframes}/{len(ds)}",flush=True)
    prior=np.concatenate(values)
    h0=float(np.median(prior))
    obj={
        "schema":SCHEMA,"partition":"train_used","dataset_root":str(Path(args.dataset_root).resolve()),
        "gt_source":"rendered_object_foreground","reference_geometry":"camera_poses + cam0_wrt_table",
        "h0_m":h0,"train_frames":nframes,"samples":int(nvalid),
        "n_train_scenes":100,"frames_per_scene":int(args.frames_per_scene),
        "max_pixels_per_frame":int(args.pixels_per_frame),"seed":args.seed,
        "statistics":{"p05":float(np.quantile(prior,.05)),
                      "median":h0,"p95":float(np.quantile(prior,.95))},
        "code_hashes":code_hashes(),"elapsed_s":perf_counter()-started,
        "no_test_or_seen_data_used":True,
    }
    atomic_json(out,obj);print(json.dumps({k:obj[k] for k in ("h0_m","train_frames","samples")},
                                   indent=2),flush=True)


def read_rows(path):
    out=[]
    if not Path(path).is_file():return out
    with open(path,encoding="utf8") as f:
        for lineno,line in enumerate(f,1):
            if line.strip():
                try: out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Incomplete/corrupt JSONL {path}:{lineno}") from exc
    return out


def rowkey(r):
    return (r["family"],int(r["epoch"]),r["split"],r["domain"],r["scene"],int(r["frame"]))


def make_manifest(args,records,expected_epochs,prior_path):
    if "train_used" not in args.splits and "train_heldout_view" not in args.splits:
        # Valid but cannot directly compute training generalization gap.
        print("[WARN] No train partition selected; P0-A gap is incomplete.",flush=True)
    return {
        "schema":SCHEMA,
        "dataset_root":str(Path(args.dataset_root).resolve()),
        "gntrans_rgb_root":str(Path(args.gntrans_rgb_root).resolve()) if args.gntrans_rgb_root else None,
        "checkpoints":records,
        "expected_epochs":expected_epochs,
        "prior_file":str(Path(prior_path).resolve()),
        "prior_sha256":sha256(prior_path),
        "splits":list(args.splits),
        "frames_per_scene":int(args.frames_per_scene),
        "max_scenes":args.max_scenes,
        "batch_size":args.batch_size,
        "appearance":args.appearance,
        "preprocess_checks_per_split":args.preprocess_checks,
        "camera":"realsense",
        "gt_source":"rendered_object_foreground",
        "crop_source":"sensor/workspace/seg; same original GraspNet crop for both appearance domains",
        "metric_masks":"common rendered fg and GT 0.2..1m and known ray dot<-0.1",
        "metrics":"exact 448x448 eligible foreground pixels; no prediction-dependent selection",
        "code_hashes":code_hashes(),
        "smoke_subsample":args.max_scenes is not None,
        "official_grasp_AP_not_run":True,
    }


def process_batch(model,record,batch,device,prior,paired):
    family=record["family"]
    pred_orig=predict(model,family,batch,device,use_trans=False)
    pred_trans=predict(model,family,batch,device,use_trans=True) if paired else None
    out=[]
    for i in range(len(batch["scene"])):
        scene=str(batch["scene"][i]);frame=int(batch["frame"][i])
        split=str(batch["split"][i])
        K=batch["K"][i].numpy()
        plane=batch["plane"][i].numpy()
        z_gt=batch["gt"][i].numpy()
        seg=batch["seg"][i].numpy()
        dot=ray_dot_from_K(K,plane,z_gt.shape)
        b=float(plane[3])
        base=dict(schema=SCHEMA,family=family,epoch=record["epoch"],
                  split=split,scene=scene,frame=frame,
                  camera_tilt_deg=float(batch["camera_tilt_deg"][i]),
                  crop_box=list(map(int,batch["crop_box"][i].tolist())))
        z=pred_orig["z"][i]
        expected_h=pred_orig["h"][i] if pred_orig["h"] is not None else None
        limits=(-.05,.35) if family=="global_film" else (
            float(model.height_min_m),float(model.height_max_m))
        met,objs,common,h_map=frame_metrics(
            z,z_gt,dot,b,seg,limits,expected_height=expected_h,prior_h=prior)
        def objinfo(items):
            return [dict(o) for o in items]
        r=dict(base,domain="original",metrics=met,objects=objinfo(objs),
               n_common_pixels=int(common.sum()),paired=None)
        out.append(r)
        if paired:
            z2=pred_trans["z"][i]
            expected_h2=pred_trans["h"][i] if pred_trans["h"] is not None else None
            met2,objs2,common2,h2=frame_metrics(
                z2,z_gt,dot,b,seg,limits,expected_height=expected_h2,prior_h=prior)
            if not np.array_equal(common,common2):
                raise ValueError("Pair did not preserve common evaluation geometry.")
            pmet=paired_appearance_metrics(z,z2,z_gt,dot,common,
                                           expected_h,expected_h2)
            pmet["input_normalized_rgb_change_l1"]=float(
                (batch["rgb"][i]-batch["rgb_trans"][i]).abs().mean())
            r["paired"]=pmet
            out.append(dict(base,domain="gntrans",metrics=met2,objects=objinfo(objs2),
                            n_common_pixels=int(common2.sum()),paired=None))
    return out


def evaluate(args):
    records,expected=read_checkpoint_registry(args.checkpoints)
    if args.appearance and not args.gntrans_rgb_root:
        raise ValueError("--appearance requires --gntrans-rgb-root.")
    h0,_=read_prior(args.prior,args.dataset_root)
    out=Path(args.output_dir)
    if out.exists() and not args.resume:
        raise FileExistsError("Output directory exists; use --resume only for exact same manifest.")
    manifest=make_manifest(args,records,expected,args.prior)
    if args.resume:
        if not out.exists():raise FileNotFoundError(out)
        existing=json.loads((out/"manifest.json").read_text())
        if existing!=manifest:
            raise ValueError("Cannot resume: checkpoint, code, data, split, or sampling protocol changed.")
    else:
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/"manifest.json",manifest)
    rows=read_rows(out/"evaluations.jsonl")
    done={rowkey(r) for r in rows}
    if len(done)!=len(rows):raise ValueError("Existing evaluation contains duplicate rows.")
    ds=build_registry(args.dataset_root,args.gntrans_rgb_root,args.splits,
                      args.frames_per_scene,args.max_scenes,
                      require_paired_rgb=args.appearance)
    # Mandatory parity checks against the existing GraspNetMultiDataset.get_data
    # before any expensive checkpoint inference; 3 deterministic positions per
    # split catch cross-scene/crop convention drift without cherry-picking.
    checks=[]
    for split in args.splits:
        matches=[i for i,task in enumerate(ds.tasks) if task[0]==split]
        if not matches:raise ValueError("No registered tasks for "+split)
        ids=np.rint(np.linspace(0,len(matches)-1,
                              min(args.preprocess_checks,len(matches)))).astype(int)
        for j in dict.fromkeys(map(int,ids)):
            checks.append(ds.verify_native_preprocessing(matches[j]))
    if not args.resume:
        atomic_json(out/"preprocess_parity.json",{"checks":checks,"n_checks":len(checks)})
    else:
        old=json.loads((out/"preprocess_parity.json").read_text())
        if old["n_checks"]!=len(checks):
            raise ValueError("Preprocessing check count changed on resume.")
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,
                      num_workers=args.workers,pin_memory=False,
                      persistent_workers=args.workers>0)
    print(f"[P0] {len(ds)} physical frames/checkpoint, models={len(records)}, "
          f"appearance_pairs={args.appearance}",flush=True)
    started=perf_counter()
    for rec in records:
        fam,epoch=rec["family"],rec["epoch"]
        requested={"original","gntrans"} if args.appearance else {"original"}
        pending_keys={
            (fam,epoch,s,domain,f"scene_{scene:04d}",frame)
            for s,_,scene,frame in ds.tasks for domain in requested
        }
        if pending_keys<=done:
            print(f"[P0] skip completed {fam}/e{epoch}",flush=True);continue
        # Checkpoint file identity checked again at model loading.
        if sha256(rec["path"])!=rec["sha256"]:
            raise ValueError("Checkpoint changed after manifest creation.")
        model,audit=load_depth_model(rec,args.device)
        atomic_json(out/f"checkpoint_{fam}_epoch{epoch}_audit.json",audit)
        # We precompute all common masks from the input only, before either model.
        with open(out/"evaluations.jsonl","a",encoding="utf8") as f:
            for bi,batch in enumerate(loader):
                # A partially completed frame can be resumed without duplicate rows:
                # compute both then append only missing per-domain records.
                new=process_batch(model,rec,batch,args.device,h0,args.appearance)
                for row in new:
                    k=rowkey(row)
                    if k in done:continue
                    f.write(json.dumps(row,allow_nan=False,sort_keys=True)+"\n")
                    done.add(k)
                f.flush()
                if bi%40==0:
                    print(f"[P0] {fam}/e{epoch}: batch={bi}/{len(loader)} "
                          f"rows={len(done)} elapsed={perf_counter()-started:.0f}s",flush=True)
        del model
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    rows=read_rows(out/"evaluations.jsonl")
    if len(rows)!=len(done):raise ValueError("Written row count != completed frame keys.")
    summarize_dir(out,rows,manifest,n_boot=args.bootstrap)
    print(f"[P0] complete: {out}",flush=True)


def write_csv(path,records):
    records=list(records)
    if not records:
        Path(path).write_text("");return
    fields=list(dict.fromkeys(k for rec in records for k in rec))
    with open(path,"w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(records)


def summarize_dir(root,rows,manifest,*,n_boot=1000):
    summary=aggregate(rows,expected_epochs=manifest["expected_epochs"],n_boot=n_boot)
    summary["manifest_sha256"]=sha256(Path(root)/"manifest.json")
    summary["raw_jsonl_sha256"]=sha256(Path(root)/"evaluations.jsonl")
    summary["n_completed_rows"]=len(rows)
    summary["n_distinct_frames"]=len({
        (r["split"],r["scene"],r["frame"]) for r in rows
    })
    summary["complete_model_runs"]={}
    # Summarization is intentionally offline: never need a mounted dataset or
    # a functioning GPU merely to regenerate aggregate tables from JSONL.
    from diagnostics.height_p0.metrics import scene_ids, deterministic_frames
    frame_count=0
    for split in manifest["splits"]:
        scenes=scene_ids(split)
        if manifest["max_scenes"] is not None:
            scenes=scenes[:int(manifest["max_scenes"])]
        frame_count+=len(scenes)*len(deterministic_frames(
            split,int(manifest["frames_per_scene"])
        ))
    for rec in manifest["checkpoints"]:
        key=f"{rec['family']}/e{rec['epoch']}"
        completed=sum(r["family"]==rec["family"] and r["epoch"]==rec["epoch"]
                      for r in rows)
        expected=frame_count*(2 if manifest["appearance"] else 1)
        summary["complete_model_runs"][key]={
            "completed_rows":completed,"expected_rows":expected,
            "complete":completed==expected
        }
    atomic_json(Path(root)/"p0_abc_summary.json",summary)

    epoch=[]
    for ident,rec in summary["metrics"].items():
        family,epoch_id,split,domain=ident.split("/")
        base=dict(family=family,epoch=int(epoch_id),split=split,domain=domain,
                  n_frames=rec["n_frames"])
        for name,stat in rec.items():
            if name=="n_frames":continue
            base[name]=stat["mean"]
            base[name+"_ci_lo"]=stat["ci95"][0] if stat["ci95"] else None
            base[name+"_ci_hi"]=stat["ci95"][1] if stat["ci95"] else None
            base["n_physical_scenes"]=stat["n_scenes"]
        epoch.append(base)
    write_csv(Path(root)/"p0a_epoch_curves.csv",epoch)
    paired=[]
    for k,v in summary["matched_epoch_contrasts"].items():
        e,split,domain,metric=k.split("/")
        paired.append(dict(epoch=int(e.replace("epoch","")),split=split,domain=domain,
                           metric=metric,mean=v["mean"],
                           ci95_low=v["ci95"][0] if v["ci95"] else None,
                           ci95_high=v["ci95"][1] if v["ci95"] else None,
                           n_scenes=v["n_scenes"]))
    write_csv(Path(root)/"p0a_matched_epoch_differences.csv",paired)
    gaps=[]
    for key,v in summary["generalization_gaps"].items():
        family,ep,domain,test_split,metric=key.split("/")
        gaps.append({"family":family,"epoch":int(ep.replace("epoch","")),
                     "domain":domain,"test_split":test_split,"metric":metric,
                     "gap_mm":v["gap"],
                     "ci95_low":v["ci95"][0],"ci95_high":v["ci95"][1],
                     "n_train_scenes":v["n_train_scenes"],
                     "n_eval_scenes":v["n_eval_scenes"]})
    write_csv(Path(root)/"p0a_generalization_gaps.csv",gaps)
    corrected=[]
    for key,v in summary["paired_generalization_gap_differences"].items():
        ep,domain,test_split,metric=key.split("/")
        corrected.append({"epoch":int(ep.replace("epoch","")),
                          "domain":domain,"test_split":test_split,
                          "metric":metric,"height_minus_pose_gap_mm":v["gap"],
                          "ci95_low":v["ci95"][0],"ci95_high":v["ci95"][1],
                          "n_train_scenes":v["n_train_scenes"],
                          "n_eval_scenes":v["n_eval_scenes"]})
    write_csv(Path(root)/"p0a_paired_gap_differences.csv",corrected)
    write_csv(Path(root)/"p0b_per_object.csv",object_pool(rows))
    app=[]
    for r in rows:
        if r.get("paired"):
            app.append({k:r[k] for k in ("family","epoch","split","scene","frame")}
                       |r["paired"])
    write_csv(Path(root)/"p0c_per_pair.csv",app)
    lines=["# Height P0-A/B/C 诊断自动汇总","","## 验证边界",
           "",f"- 记录行数：{len(rows)}；物理帧数：{summary['n_distinct_frames']}。",
           "- **不是Grasp AP**，只计算DPT的固定裁剪与共同前景几何指标。",
           "- GraspNet sensor depth和seg用于构建原版workspace/crop；仅以rendered前景深度为监督参考。",
           "- 外部camera/table标定由GraspNet metadata提供，属于特权几何输入。",
           "- Train-heldout-view是训练物体的未用视角，不能当成未见物体。",
           "- 缺少对应epoch权重时不能声称已验证过拟合曲线。",
           "- 配对外观实验的两个RGB共用完全相同的crop、K、plane和GT；GN-Trans可能同时改变材质与光照。",
           "", "## Checkpoint完整性"]
    for k,v in summary["complete_model_runs"].items():
        lines.append(f"- {k}: {v['completed_rows']}/{v['expected_rows']} rows；complete={v['complete']}")
    lines+=["","## P0-A 已有与缺失 epoch"]
    for fam,eps in summary["available_epochs"].items():
        lines.append(f"- {fam}: {eps}; 缺少预声明轮次 {summary['missing_expected_epochs'].get(fam,[])}")
    lines+=["","## P0-A 过拟合曲线可识别性",
            f"- 两架构拥有至少两个共同epoch：{summary['trajectory_eligible']['matched_both_families_at_least_two_epochs']}。",
            "- 单个完成轮次只能说明cross-sectional误差差距，不能检验过拟合是否随训练发展。",
            "","## P0-A/B/C 产物",
            "- `p0a_epoch_curves.csv`：按scene均权的Depth/Height曲线。",
            "- `p0a_matched_epoch_differences.csv`：严格相同scene/frame的跨模型配对差值及scene bootstrap CI。",
            "- `p0a_generalization_gaps.csv`：每模型的Test-Train scene-balanced泛化差距。",
            "- `p0a_paired_gap_differences.csv`：同epoch Height-Pose的泛化差距增量。",
            "- `p0b_per_object.csv`：实例高度均值、方差、GT-oracle常数高度、空间R²和偏差。",
            "- `p0c_per_pair.csv`：相同几何下GNTrans-RS的预测变化和对GT误差变化。",
            "- `p0_abc_summary.json`：原始汇总与缺失检查。",
            "- `evaluations.jsonl`：逐帧可审计结果。",
            ""]
    (Path(root)/"REPORT_ZH.md").write_text("\n".join(lines),encoding="utf8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("fit-prior")
    a.add_argument("--dataset-root",required=True)
    a.add_argument("--output",required=True)
    a.add_argument("--frames-per-scene",type=int,default=16)
    a.add_argument("--pixels-per-frame",type=int,default=2048)
    a.add_argument("--seed",type=int,default=0)
    e=sub.add_parser("evaluate")
    e.add_argument("--dataset-root",required=True)
    e.add_argument("--gntrans-rgb-root")
    e.add_argument("--checkpoints",required=True)
    e.add_argument("--prior",required=True)
    e.add_argument("--output-dir",required=True)
    e.add_argument("--splits",nargs="+",choices=SPLITS,default=list(SPLITS))
    e.add_argument("--frames-per-scene",type=int,default=8)
    e.add_argument("--max-scenes",type=int)
    e.add_argument("--workers",type=int,default=2)
    e.add_argument("--batch-size",type=int,default=2)
    e.add_argument("--preprocess-checks",type=int,default=3,
                   help="Required deterministic native data-loader parity checks per split.")
    e.add_argument("--bootstrap",type=int,default=1000)
    e.add_argument("--device",default="cuda:0")
    e.add_argument("--appearance",action="store_true")
    e.add_argument("--resume",action="store_true")
    s=sub.add_parser("summarize");s.add_argument("--output-dir",required=True)
    s.add_argument("--bootstrap",type=int,default=1000)
    args=p.parse_args()
    if args.command=="fit-prior":
        if args.pixels_per_frame<1:raise ValueError("pixels_per_frame must be >0")
        fit_prior(args)
    elif args.command=="evaluate":
        if args.batch_size<1 or args.workers<0 or args.bootstrap<10 or args.preprocess_checks<1:
            raise ValueError("Invalid batch/worker/bootstrap.")
        evaluate(args)
    else:
        out=Path(args.output_dir)
        summary=read_rows(out/"evaluations.jsonl")
        if not summary:raise ValueError("No evaluated frames.")
        manifest=json.loads((out/"manifest.json").read_text())
        summarize_dir(out,summary,manifest,n_boot=args.bootstrap)
        print(out/"REPORT_ZH.md")


if __name__=="__main__":
    main()
