#!/usr/bin/env python3
"""P1-1: decompose GC6D positive-label scarcity and depth-center mismatch.

Offline: raw per-object friction/width/collision and scene point-retention audit.
Counterfactual: same checkpoint and RGB input, compare predicted-depth versus
project-generated GT-depth matching. GT intervention is PRIVILEGED DIAGNOSTIC,
not a deployable RGB-only result. No model weights are modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from mixed_p1_historical import (
    activate_historical, load_model, make_train_sources, sample_source_indices,
)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("offline","counterfactual","both"),default="offline")
    p.add_argument("--gc6d-root",required=True)
    p.add_argument("--filtered-scenes",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--historical-root")
    p.add_argument("--checkpoint")
    p.add_argument("--graspnet-root")
    p.add_argument("--gntrans-root")
    p.add_argument("--raw-max-scenes",type=int,default=24,
                   help="0 scans 232 scenes; defaults 24 for massive raw grasp tensors")
    p.add_argument("--forward-frames",type=int,default=24)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--point-filter-mu",type=float,default=.4)
    p.add_argument("--max-width",type=float,default=.1)
    p.add_argument("--invalid-gt-policy",choices=("keep_pred","zero"),default="keep_pred")
    return p.parse_args()


def write_csv(file,rows):
    if not rows:return
    file=Path(file);file.parent.mkdir(parents=True,exist_ok=True)
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with file.open("w",newline="",encoding="utf-8") as fd:
        w=csv.DictWriter(fd,fieldnames=cols);w.writeheader();w.writerows(rows)


def read_scene_ids(file):
    ids=json.loads(Path(file).read_text(encoding="utf-8"))
    if not isinstance(ids,list) or len(ids)!=232 or len(set(map(int,ids)))!=232:
        raise ValueError("Expected filtered GC6D JSON array with 232 unique scenes")
    return sorted(map(int,ids))


def raw_object_stats(fric,width,collision,point_filter_mu,max_width):
    if fric.ndim!=4 or fric.shape!=width.shape or fric.shape!=collision.shape:
        raise ValueError("Object arrays need identical [P,V,A,D] shapes")
    if not fric.shape[0]:
        raise ValueError("No object grasp points")
    finite_fric=np.isfinite(fric)
    physical_fric=finite_fric&(fric>0)&(fric<=1.2)
    width_ok=(width>=0)&(width<=max_width)&np.isfinite(width)
    col=np.asarray(collision,bool)
    success=physical_fric&width_ok&~col
    parts=[
        int((~physical_fric).sum()),
        int((physical_fric&~width_ok).sum()),
        int((physical_fric&width_ok&col).sum()),
        int(success.sum()),
    ]
    if sum(parts)!=fric.size:
        raise AssertionError("Failure partition does not cover all candidate labels")
    P=fric.shape[0]
    # Exact GC6D cache-generation point filter: width is deliberately NOT
    # part of this stage, whereas CDF success requires a valid width.
    keep=np.any(((fric>0)&(fric<=point_filter_mu)&finite_fric&~col).reshape(P,-1),axis=1)
    valid_obj=np.any(success.reshape(P,-1),axis=1)
    return dict(
        raw_points=int(P),raw_candidates=int(fric.size),
        bad_friction_candidates=parts[0],bad_width_given_friction=parts[1],
        collision_given_friction_width=parts[2],
        successful_candidates=parts[3],
        positive_candidate_fraction=float(parts[3]/fric.size),
        collision_fraction=float(col.mean()),
        kept_points_expected=int(keep.sum()),
        physically_graspable_points=int(valid_obj.sum()),
        graspable_points_lost_by_04_filter=int((valid_obj&~keep).sum()),
        raw_point_retention_fraction=float(keep.mean()),
    )


def run_offline(a,ids,out):
    root=Path(a.gc6d_root).resolve()
    scenes=ids[:a.raw_max_scenes] if a.raw_max_scenes else ids
    rows=[];fail=[]
    for index,sid in enumerate(scenes):
        sid6=f"{sid:06d}"
        cached=root/"economic_grasp_label_300views_extend_angle_cdf_depth"/f"{sid6}_labels.npz"
        if not cached.exists():raise FileNotFoundError(cached)
        with np.load(cached,allow_pickle=False) as d:
            objids=np.asarray(d["obj_ids"],int)
            collidx=np.asarray(d["collision_arr_indices"],int)
            pointid=np.asarray(d["pointid"],int)
            if len(objids)!=len(collidx) or (len(pointid) and pointid.max()>=len(objids)):
                raise ValueError(f"Invalid object-ID cache mapping at {cached}")
        collision_file=root/"collision_label"/f"{sid6}.npz"
        with np.load(collision_file,allow_pickle=False) as collisions:
            for local_i,objid in enumerate(objids):
                filename=root/"grasp_label"/f"obj_{objid:06d}_labels.npz"
                with np.load(filename,allow_pickle=False) as d:
                    fric=np.asarray(d["scores"],np.float32)
                    width=np.asarray(d["offsets"][...,2],np.float32)
                ck=f"arr_{int(collidx[local_i])}"
                if ck not in collisions:raise KeyError(f"{collision_file}: {ck} missing")
                col=np.asarray(collisions[ck],bool)
                st=raw_object_stats(fric,width,col,a.point_filter_mu,a.max_width)
                cached_retained=int((pointid==local_i).sum())
                row=dict(scene=sid,obj_id=int(objid),local_object=int(local_i),
                         cached_retained_points=cached_retained,**st)
                row["cache_retention_agrees"]=cached_retained==st["kept_points_expected"]
                if not row["cache_retention_agrees"]:
                    fail.append(row)
                rows.append(row)
                del fric,width,col
        if (index+1)%5==0:
            print(f"[P1-1 raw] {index+1}/{len(scenes)} scenes",flush=True)
    write_csv(out/"gc6d_raw_label_decomposition.csv",rows)
    if fail:
        write_csv(out/"retention_disagreements.csv",fail)
        raise RuntimeError(
            f"Cache raw point retention differs in {len(fail)} objects. "
            "Check generator parameters, object mapping and label versions."
        )
    keys=("raw_points","raw_candidates","bad_friction_candidates",
          "bad_width_given_friction","collision_given_friction_width",
          "successful_candidates","kept_points_expected",
          "physically_graspable_points","graspable_points_lost_by_04_filter")
    sums={k:sum(x[k] for x in rows) for k in keys}
    sums["success_candidate_fraction"]=sums["successful_candidates"]/max(sums["raw_candidates"],1)
    sums["graspable_point_filter_loss_fraction"]=sums["graspable_points_lost_by_04_filter"]/max(
        sums["physically_graspable_points"],1)
    return dict(scenes=len(scenes),objects=len(rows),complete_scenes=len(scenes)==232,
                stage_counts=sums,original_point_filter_mu=a.point_filter_mu,
                caveat="Raw synthetic analytic success labels, not demonstrated physical robot execution.")


def frame_summary(ep):
    valid=ep["batch_grasp_cdf_valid_mask"].detach().bool()
    bins=ep["batch_grasp_cdf_bins_angle_depth"].detach().long()
    if valid.ndim!=4 or bins.shape!=valid.shape:
        raise ValueError(f"Bad matched label shapes: {valid.shape} {bins.shape}")
    q=valid.any(dim=(-1,-2))
    pos=valid&(bins>0)
    return dict(
        queries=int(q.numel()),matched_queries=int(q.sum()),
        matched_query_rate=float(q.float().mean()),
        matched_actions=int(valid.sum()),positive_actions=int(pos.sum()),
        positive_queries=int(pos.any(dim=(-1,-2)).sum()),
        zero_positive_frame=not bool(pos.any()),
        positive_given_matched=float(pos.sum()/max(int(valid.sum()),1)))


def depth_counterfactual(model,batch,invalid_policy):
    import torch.nn.functional as F
    with torch.no_grad():
        original=model(dict(batch))
    gt=batch["gt_depth_m"].detach().float()
    if gt.ndim==3:gt=gt[:,None]
    mask=torch.isfinite(gt)&(gt>=.2)&(gt<=1.0)
    if not bool(mask.any()):
        raise ValueError("GC6D sample has no supervised GT-depth pixels")
    def gt_override(_module,_args,output):
        result=list(output)
        live=result[0]
        if gt.shape!=live.shape:
            raise ValueError(f"GT and DPT output dimensions differ: {gt.shape}/{live.shape}")
        new=torch.where(mask,gt.to(live),live if invalid_policy=="keep_pred" else torch.zeros_like(live))
        result[0]=new
        result[1]=F.interpolate(new,size=result[1].shape[-2:],mode="nearest")
        return tuple(result)
    handle=model.base.depth_net.register_forward_hook(gt_override)
    try:
        with torch.no_grad():
            oracle=model(dict(batch))
    finally:
        handle.remove()
    return original,oracle,float(mask.float().mean())


def nearest_center_distance_mm(a,b):
    from scipy.spatial import cKDTree
    if len(a)==0 or len(b)==0:return None
    da,_=cKDTree(np.asarray(b).reshape(-1,3)).query(np.asarray(a).reshape(-1,3),k=1)
    return dict(mean_mm=float(da.mean()*1000),p90_mm=float(np.percentile(da,90)*1000))


def run_counterfactual(a,out):
    required=(a.historical_root,a.checkpoint,a.graspnet_root,a.gntrans_root)
    if not all(required):
        raise ValueError("--historical-root --checkpoint --graspnet-root --gntrans-root required")
    if not torch.cuda.is_available():
        raise RuntimeError("A single CUDA GPU is required")
    activate_historical(a.historical_root)
    model,ck,meta=load_model(a.historical_root,a.checkpoint,"cuda:0")
    from moge_rayrope.runtime import move_batch
    from dataset.graspnet_dataset import collate_fn
    sources=make_train_sources(a.graspnet_root,a.gntrans_root,a.gc6d_root,
                               a.filtered_scenes,model.cfg)
    ds=sources["gc6d"][0]
    indexes=sample_source_indices(ds,a.forward_frames,a.seed)
    rows=[]
    for row,idx in enumerate(indexes):
        np.random.seed(a.seed+idx)
        torch.manual_seed(a.seed+idx)
        batch=move_batch(collate_fn([ds[idx]]),"cuda:0",labels=True)
        p,o,valid_gt=depth_counterfactual(model,batch,a.invalid_gt_policy)
        fp,fo=frame_summary(p),frame_summary(o)
        drift=nearest_center_distance_mm(
            p["xyz_graspable"].detach().cpu().numpy(),
            o["xyz_graspable"].detach().cpu().numpy())
        rows.append(dict(scene=str(ds.scenename[idx]),frame=int(ds.frameid[idx]),
                         dataset_index=idx,gt_depth_valid_fraction=valid_gt,
                         **{f"pred_{k}":v for k,v in fp.items()},
                         **{f"target_{k}":v for k,v in fo.items()},
                         delta_match_rate=fo["matched_query_rate"]-fp["matched_query_rate"],
                         delta_positive_actions=fo["positive_actions"]-fp["positive_actions"],
                         center_nearest_mean_mm=drift["mean_mm"] if drift else None,
                         center_nearest_p90_mm=drift["p90_mm"] if drift else None))
        print(f"[P1-1 geometry] {row+1}/{len(indexes)} pred={fp['matched_query_rate']:.3f} target={fo['matched_query_rate']:.3f}",flush=True)
    write_csv(out/"gc6d_pred_vs_target_geometry.csv",rows)
    def avg(k):return float(np.mean([float(x[k]) for x in rows]))
    return dict(checkpoint_sha256=meta["checkpoint_sha256"],variant=meta["variant"],
        frames=len(rows),avg_pred_match_rate=avg("pred_matched_query_rate"),
        avg_target_match_rate=avg("target_matched_query_rate"),
        avg_delta_match_rate=avg("delta_match_rate"),
        avg_delta_positive_actions=avg("delta_positive_actions"),
        zero_positive_pred_frames=sum(x["pred_zero_positive_frame"] for x in rows),
        zero_positive_target_frames=sum(x["target_zero_positive_frame"] for x in rows),
        mean_nearest_center_drift_mm=avg("center_nearest_mean_mm"),
        invalid_gt_policy=a.invalid_gt_policy,
        caveat="Privileged target-depth intervention changes geometry/seed proposals and label matching, "
               "not merely confidence. It is not an attainable RGB-only score.")


def main():
    a=parse_args()
    if a.raw_max_scenes<0 or a.forward_frames<1 or not 0<a.point_filter_mu<=1.2 or a.max_width<=0:
        raise ValueError("Invalid raw scene/frame or label filter settings")
    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    ids=read_scene_ids(a.filtered_scenes)
    report=dict(status="complete",args=vars(a))
    if a.mode in ("offline","both"):
        report["raw_label_decomposition"]=run_offline(a,ids,out)
    if a.mode in ("counterfactual","both"):
        report["geometry_counterfactual"]=run_counterfactual(a,out)
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(status="complete",summary=str(out/"summary.json")),indent=2))


if __name__=="__main__":
    main()
