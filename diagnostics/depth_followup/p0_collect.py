"""P0 collect train Original/GNTrans vs GC6D metric-distance/crop/K data.

Same frozen MIX50 Point-GVAR e15 depth_net and the native crop policy for
each domain; 100 GraspNet train scenes × fixed 4 frames/domain, 78 GC6D
scenes × 13 frames. GNTrans reference is virtual depth, NOT sensor depth.
This is observational distribution analysis, never a calibration fit.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from diagnostics.frozen_depth_response.geometry import paired_depth_metrics,HW
from .common import (check_checkpoint, digest, frozen_depth, gc6d_frame,
                     graspnet_frame, sha256, write_json)


def _scenes(path):
    values=json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(values,list) or not values or
            any(type(i)!=int or i<0 for i in values) or
            len(set(values))!=len(values)):
        raise ValueError("GC6D scene IDs require unique nonnegative integer JSON array")
    return sorted(values)


def build_dataset(a):
    if a.domain=="gc6d":
        from dataset.gc6d_dataset import GraspClutter6DMultiDataset
        ids=_scenes(a.gc6d_scenes)
        ds=GraspClutter6DMultiDataset(
            root=a.gc6d_root,camera="realsense-d435",split=a.gc6d_split,
            scene_ids=ids,max_ann=13,mask_mode="workspace_depth",
            workspace_pose_mode="json",workspace_outlier=.02,
            workspace_depth_trunc=0.,factor_depth_mode="bop",
            load_label=False,augment=False,load_grasp_payload=False,
            use_gt_depth=False,use_fuse_depth=False,
            fallback_to_depth_mask_on_empty=False)
        selected=list(range(len(ds)))
        if not a.smoke and (len(ids)!=78 or len(ds)!=1014 or
                            len({int(s) for s,_,_ in ds.samples})!=78):
            raise ValueError("Formal P0 GC6D requires accepted 78 scenes/1014 frames")
    else:
        from dataset.graspnet_dataset import GraspNetMultiDataset,GraspNetTransDataset
        annos=[int(x) for x in a.train_annos.split(",")]
        if len(annos)!=len(set(annos)) or len(annos)!=4 or any(
                x<0 or x>=256 or x%2 for x in annos):
            raise ValueError("Primary MIX50 train-domain sample: four distinct even annos")
        if a.domain=="original":
            ds=GraspNetMultiDataset(
                a.graspnet_root,camera="realsense",split="train",
                remove_outlier=True,load_label=False,augment=False,
                use_gt_depth=False,use_fuse_depth=False,depth_strides=1)
        elif a.domain=="gntrans":
            ds=GraspNetTransDataset(
                a.graspnet_root,a.gntrans_root,camera="realsense",
                split="train",remove_outlier=True,load_label=False,
                augment=False,use_gt_depth=True,depth_strides=1)
        else:raise ValueError("Unknown data domain")
        selected=[i for i,(s,anno) in enumerate(zip(ds.scenename,ds.frameid))
                  if int(anno) in annos]
        if not a.smoke and (len(selected)!=400 or
              {int(str(ds.scenename[i]).split("_")[-1]) for i in selected}!=set(range(100))):
            raise ValueError("Formal P0 GraspNet source coverage must be 100 scenes × 4")
    if a.max_frames:
        if not a.smoke or a.max_frames>=len(selected):
            raise ValueError("Nonzero --max-frames only for independent smoke")
        selected=selected[:a.max_frames]
    return ds,selected


def _load_reused(manifest_file,frame):
    """Optional strict P1 GC6D cache reuse to avoid redundant frozen forward."""
    mf=json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    if (mf.get("schema")!="gc6d-surface-depth-cache-v1" or
            mf.get("base_checkpoint_sha256")!=frame["checkpoint_sha256"]):
        raise ValueError("P1 surface cache incompatible with frozen checkpoint")
    rows=mf.get("records",[])
    idx={(int(r["scene"]),int(r["img_id"])):r for r in rows}
    key=(frame["scene"],frame["img_id"])
    if key not in idx:raise ValueError(f"Missing cache frame {key}")
    r=idx[key]
    if (r["source_sha256"]["rgb"]!=frame["source_sha256"]["rgb"]
            or r["source_sha256"]["depth"]!=frame["source_sha256"]["depth"]
            or r["source_sha256"]["seg"]!=frame["source_sha256"]["seg"]
            or not np.array_equal(np.asarray(r["K"],dtype=np.float32),frame["K"])
            or not np.array_equal(np.asarray(r["crop_box"],dtype=np.int32),
                                  frame["crop_box"])):
        raise RuntimeError(f"P1 cached metric-depth frame source/crop changed: {key}")
    path=Path(r["prediction_file"])
    if not path.is_file() or sha256(path)!=r["prediction_sha256"]:
        raise RuntimeError(f"P1 prediction bytes modified: {path}")
    with np.load(path,allow_pickle=False) as z:
        p=z["pred_depth"]
    if p.shape!=HW or not np.isfinite(p).all():
        raise RuntimeError("Invalid cached metric depth")
    return p


def _sample_stats(frame,pred,a):
    mask=np.asarray(frame["foreground"],bool)
    ref=np.asarray(frame["reference"],dtype=np.float32)
    met=paired_depth_metrics(pred,ref,mask,a.min_pixels)
    if not met["valid"]:
        raise RuntimeError(f"Insufficient reference foreground: {frame['domain']}/"
                           f"{frame['scene']}/{frame['anno']} count={met['valid_pixels']}")
    h,w=frame["raw_hw"]
    ch,cw=frame["crop_hw"]
    k=frame["K"];raw=frame["raw_K"]
    obs=ref[mask&(ref>0)]
    if not len(obs):raise RuntimeError("Observed mask empty")
    row=dict(
        domain=frame["domain"],reference_source=frame["reference_source"],
        scene=frame["scene"],img_id=frame["img_id"],anno=frame["anno"],
        raw_h=h,raw_w=w,crop_h=ch,crop_w=cw,
        crop_aspect=float(cw/ch),
        crop_area_fraction=frame["crop_area_fraction"],
        crop_fraction_w=float(cw/w),crop_fraction_h=float(ch/h),
        foreground_fraction=frame["foreground_fraction"],
        foreground_full_pixels=frame["foreground_full_pixels"],
        native_mask_pixels=frame["native_mask_pixels"],
        crop_foreground_fraction=float(mask.mean()),
        fx_raw=float(raw[0,0]),fy_raw=float(raw[1,1]),
        fx_crop=float(k[0,0]),fy_crop=float(k[1,1]),
        cx_crop=float(k[0,2]),cy_crop=float(k[1,2]),
        focal_ratio_x=float(k[0,0]/raw[0,0]),
        focal_ratio_y=float(k[1,1]/raw[1,1]),
        focal_mean_crop=float(np.sqrt(k[0,0]*k[1,1])),
        foreground_reference_p10=float(np.quantile(obs,.1)),
        foreground_reference_p90=float(np.quantile(obs,.9)),
        valid_pixels=met["valid_pixels"],
        median_pred=met["median_pred"],
        median_reference=met["median_reference"],
        median_ratio=met["median_ratio"],
        median_error=met["median_error"],
        mae=met["mae"],absrel=met["absrel"],
        camera_pose_z=float(frame["pose"][2]),
        camera_pose_tilt_deg=float(np.degrees(np.arccos(
            np.clip(float(frame["pose"][2]),-1.,1.)))),
        crop_K_max_error=frame["crop_K_error"],
        source_sha256=digest(frame["source_sha256"]),
    )
    return row


def run(a):
    if a.min_pixels<32 or a.max_frames<0:raise ValueError("Invalid P0 sampling")
    ds,selected=build_dataset(a)
    from rare_grasp.runtime import Runtime
    rt=Runtime(a.bound_config,a.graspnet_root,a.gntrans_root,a.device)
    model=check_checkpoint(rt)
    cache_manifest=(a.reuse_gc6d_cache if a.domain=="gc6d" else None)
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use new P0 output directory; never overwrite accepted GC6D")
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,idx in enumerate(selected,1):
        frame=(gc6d_frame(ds,idx) if a.domain=="gc6d" else
               graspnet_frame(ds,idx,domain=a.domain))
        frame["checkpoint_sha256"]=rt.config["checkpoint_sha256"]
        pred=(_load_reused(cache_manifest,frame) if cache_manifest else
              frozen_depth(model,frame,rt.device))
        row=_sample_stats(frame,pred,a)
        rows.append(row)
        if i%25==0 or i==len(selected):
            print(f"[P0 {a.domain}] {i}/{len(selected)} "
                  f"scene={frame['scene']} image={frame['img_id']}",flush=True)
    csv_path=root/"per_frame.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    meta=dict(schema="frozen-depth-p0-domain-v1",
        kind="native-crop-distance-and-focal-statistics",domain=a.domain,
        complete=not a.smoke and not a.max_frames,
        frames=len(rows),
        scenes=len({r["scene"] for r in rows}),
        base_checkpoint_sha256=rt.config["checkpoint_sha256"],
        base_config_digest=rt.config_digest,
        bound_config_sha256=sha256(a.bound_config),
        source_dataset_root=str(Path(a.gc6d_root if a.domain=="gc6d"
                                  else a.graspnet_root).resolve()),
        gntrans_root=str(Path(a.gntrans_root).resolve()) if a.domain=="gntrans" else None,
        scene_list_sha256=sha256(a.gc6d_scenes) if a.domain=="gc6d" else None,
        train_annotations=a.train_annos if a.domain!="gc6d" else None,
        input_crop=("GC6D workspace_depth native" if a.domain=="gc6d"
                    else "GraspNet original workspace mask native"),
        reference=("GC6D/Original sensor" if a.domain!="gntrans"
                   else "GNTrans virtual renderer, NOT physical sensor"),
        input_privilege="Sensor depth/segmentation determine crop, never passed to depth head",
        reference_only_for_offline_evaluation=True,
        reused_gc6d_cache_sha256=sha256(cache_manifest) if cache_manifest else None,
        dataset_source_code_sha256=sha256(
            Path("dataset")/("gc6d_dataset.py" if a.domain=="gc6d" else "graspnet_dataset.py")),
        data_adapter_code_sha256=sha256(Path(__file__).with_name("common.py")),
        collector_code_sha256=sha256(__file__),
        per_frame_sha256=sha256(csv_path),train_updates=0,grasp_AP_evaluations=0,
        limitations=[
            "GC6D scenes are previously inspected exploratory data",
            "Original/GNTrans train scenes were seen by the frozen base",
            "GNTrans virtual depth and Original/GC6D sensor reference have different provenance",
            "Foreground mask/source crop geometry can be different across domains",
            "Not a physical focal length intervention or metric-depth scale fit",
        ])
    write_json(root/"manifest.json",meta)
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain",choices=("original","gntrans","gc6d"),required=True)
    p.add_argument("--bound-config",required=True)
    p.add_argument("--graspnet-root",required=True)
    p.add_argument("--gntrans-root",default="")
    p.add_argument("--gc6d-root",default="")
    p.add_argument("--gc6d-scenes",default="")
    p.add_argument("--gc6d-split",default="test")
    p.add_argument("--train-annos",default="0,64,128,192")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--min-pixels",type=int,default=128)
    p.add_argument("--reuse-gc6d-cache",default=None)
    p.add_argument("--max-frames",type=int,default=0)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    if a.domain=="gntrans" and not a.gntrans_root:
        p.error("GNTrans RGB root required")
    if a.domain=="gc6d" and (not a.gc6d_root or not a.gc6d_scenes):
        p.error("GC6D dataset root and accepted scene IDs JSON required")
    run(a)


if __name__=="__main__":
    main()
