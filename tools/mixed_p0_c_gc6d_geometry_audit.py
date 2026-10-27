#!/usr/bin/env python3
"""P0-C: GC6D sensor/virtual depth disagreement under actual training crop.

This is an *audit*, not a change to depth supervision or inference. Also
compares BOP scene_camera rotation interpretations ('as C2W' vs 'as W2C') via
multi-frame object pose consistency. The world +Z up assumption remains
explicit; pose consistency alone cannot determine physical vertical.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from dataset.gc6d_dataset import (
    GraspClutter6DMultiDataset, build_mask, depth_to_cloud,
    get_factor_depth, load_json, load_label, pose_from_gc6d_obj,
)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gc6d-root", required=True)
    p.add_argument("--filtered-scenes", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--camera", default="realsense-d435")
    p.add_argument("--max-ann", type=int, default=13)
    p.add_argument("--frame-stride", type=int, default=20,
                   help="Use flattened training indices 0,20,... by default; 1 means all frames.")
    p.add_argument("--pose-scene-limit", type=int, default=0,
                   help="0 means all scenes for pose consistency.")
    p.add_argument("--depth-min", type=float, default=.2)
    p.add_argument("--depth-max", type=float, default=1.0)
    p.add_argument("--workspace-pose-mode", choices=("json","inverse","none"), default="json")
    p.add_argument("--workspace-outlier", type=float, default=.02)
    p.add_argument("--factor-depth-mode", choices=("bop","camera","fixed"), default="bop")
    p.add_argument("--fixed-factor-depth", type=float, default=None)
    p.add_argument("--outlier-threshold-mm", type=float, default=50)
    return p.parse_args()


def save_csv(file, rows):
    if not rows:
        return
    cols = list(rows[0])
    with Path(file).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def summary(vals):
    v = np.asarray(vals, np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return dict(count=int(len(v)),mean=float(v.mean()),median=float(np.median(v)),
                p90=float(np.percentile(v,90)),p95=float(np.percentile(v,95)),
                maximum=float(np.max(v)))


def pose_matrix_camera(cam_info):
    t = np.eye(4, dtype=np.float64)
    t[:3,:3] = np.asarray(cam_info["cam_R_w2c"],np.float64).reshape(3,3)
    t[:3,3] = np.asarray(cam_info["cam_t_w2c"],np.float64).reshape(3)/1000.0
    return t


def pose_matrix_object(x):
    return pose_from_gc6d_obj(x).astype(np.float64)


def directional_deg(u, v):
    u=np.asarray(u,float); v=np.asarray(v,float)
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/
         (np.linalg.norm(u)*np.linalg.norm(v)+1e-12),-1.0,1.0))))


def compare_pose_hypotheses(root, dataset, max_scenes):
    """World-stationary object origins under camera-as-C2W vs its inverse.

    For repeated category IDs, frame-to-frame assignment uses Hungarian
    matching in each hypothesis to avoid treating duplicate IDs as one instance.
    Scene GT is assumed static for this diagnostic; moving objects are excluded
    only if the user's dataset-specific annotations identify them separately.
    """
    metrics=[]
    for scene_id in dataset.scene_ids[:max_scenes or None]:
        scene_id=int(scene_id)
        cam=load_json(root/"scenes"/f"{scene_id:06d}"/"scene_camera.json")
        objects=load_json(root/"scenes"/f"{scene_id:06d}"/"scene_gt.json")
        frame_list=[(int(img),int(ann)) for sid,img,ann in dataset.samples if int(sid)==scene_id]
        frame_list.sort(key=lambda x:x[1])
        if len(frame_list)<2:
            continue
        for hypothesis in ("camera_json_is_c2w","camera_json_is_w2c"):
            anchor={}
            for fi,(img_id,ann_id) in enumerate(frame_list):
                caminfo=cam[str(img_id)]
                supplied=pose_matrix_camera(caminfo)
                world_from_camera = supplied if hypothesis=="camera_json_is_c2w" else np.linalg.inv(supplied)
                by_obj=defaultdict(list)
                for obj in objects[str(img_id)]:
                    Tw_o=world_from_camera @ pose_matrix_object(obj)
                    by_obj[int(obj["obj_id"])].append(Tw_o[:3,3])
                if fi==0:
                    anchor={k:np.stack(v) for k,v in by_obj.items()}
                    continue
                for objid,curr in by_obj.items():
                    if objid not in anchor:
                        continue
                    old=anchor[objid]; current=np.stack(curr)
                    cost=np.linalg.norm(old[:,None,:]-current[None,:,:],axis=-1)
                    ref_i,cur_i=linear_sum_assignment(cost)
                    for i,j in zip(ref_i,cur_i):
                        metrics.append(dict(scene_id=scene_id,hypothesis=hypothesis,
                            img_id=img_id,ann_id=ann_id,obj_id=objid,
                            displacement_mm=float(1000*cost[i,j]),
                            num_same_id_instances=len(current)))
    return metrics


def build_depth_frame(dataset, i, a):
    sid,img_id,ann=dataset.samples[i]
    caminfo = dataset._load_scene_camera(sid)[str(int(img_id))]
    K=np.asarray(caminfo["cam_K"],np.float32).reshape(3,3)
    factor=get_factor_depth(dataset.camera,caminfo,a.factor_depth_mode,a.fixed_factor_depth)
    raw=np.asarray(Image.open(dataset.depthpath[i]),np.float32)
    virtual=np.asarray(Image.open(dataset.gtdepthpath[i]),np.float32)
    seg=load_label(dataset.labelpath[i])
    if raw.shape!=virtual.shape or raw.shape!=seg.shape:
        raise ValueError(f"GC6D RGB/depth/label misaligned at {sid}/{img_id}")
    cloud=depth_to_cloud(raw,K,factor)
    mask=build_mask(raw,seg,cloud,caminfo,"workspace_depth",
                    workspace_outlier=a.workspace_outlier,
                    workspace_pose_mode=a.workspace_pose_mode)
    if not mask.any():
        raise ValueError(f"Empty training workspace mask at {sid}/{img_id}")
    x0,y0,x1,y1=dataset._crop_box_from_mask(mask)
    if x0>=x1 or y0>=y1:
        raise ValueError(f"Invalid crop for GC6D scene {sid} image {img_id}")
    def resize(src):
        return cv2.resize(src[y0:y1,x0:x1],(448,448),interpolation=cv2.INTER_NEAREST)
    observed=resize(raw/factor)
    target=resize(virtual/factor)
    lbl=resize(seg)
    # Actual compute_depth_reg_loss uses valid GT pixels at [min,max];
    # no extra workspace loss-mask beyond the crop.
    target_valid=(target>0)&(target>=a.depth_min)&(target<=a.depth_max)
    obs_valid=(observed>0)&np.isfinite(observed)
    both=target_valid & obs_valid
    result=dict(dataset_index=i,scene_id=int(sid),img_id=int(img_id),ann_id=int(ann),
       original_height=int(raw.shape[0]),original_width=int(raw.shape[1]),
       crop_xyxy=f"{x0},{y0},{x1},{y1}",
       crop_area_ratio=float(((x1-x0)*(y1-y0))/raw.size),
       depth_target_valid_ratio=float(target_valid.mean()),
       observed_valid_ratio=float(obs_valid.mean()),
       both_valid_ratio=float(both.mean()),
       target_only_ratio=float((target_valid&~obs_valid).mean()),
       observed_only_ratio=float((obs_valid&~target_valid).mean()))
    for label,region in (("foreground",lbl>0),("background",lbl==0),("all",np.ones_like(lbl,bool))):
        valid=both & region
        n=int(valid.sum())
        result[f"{label}_joint_valid_pixels"]=n
        result[f"{label}_joint_valid_ratio"]=float(n/max(int(region.sum()),1))
        if n:
            error=np.abs(observed[valid]-target[valid])*1000.0
            result[f"{label}_mae_mm"]=float(error.mean())
            result[f"{label}_p95_mm"]=float(np.percentile(error,95))
            result[f"{label}_gt50mm_ratio"]=float(np.mean(error>a.outlier_threshold_mm))
            result[f"{label}_total_abs_error_mm"]=float(error.sum())
        else:
            result[f"{label}_mae_mm"]=None
            result[f"{label}_p95_mm"]=None
            result[f"{label}_gt50mm_ratio"]=None
            result[f"{label}_total_abs_error_mm"]=0.0
    return result


def audit_camera_vectors(dataset):
    """Compare existing GC6D dataset FiLM input with C2W-consistent inputs."""
    rows=[]
    for idx in range(len(dataset)):
        sid,img_id,ann_id=dataset.samples[idx]
        meta=dataset._load_scene_camera(sid)[str(img_id)]
        R=np.asarray(meta["cam_R_w2c"],np.float64).reshape(3,3)
        legacy_pose,legacy_up,legacy_tilt=dataset.build_camera_orientation_metadata(meta)
        corrected_pose= -R[:,2] # if supplied matrix actually C2W
        corrected_up= R.T @ np.array([0.,0.,1.])
        corrected_tilt=float(np.degrees(np.arccos(np.clip(corrected_pose[2],-1.,1.))))
        rows.append(dict(scene_id=int(sid), img_id=int(img_id),ann_id=int(ann_id),
           legacy_tilt_deg=float(legacy_tilt),c2w_tilt_deg=corrected_tilt,
           pose_vector_angle_delta_deg=directional_deg(legacy_pose,corrected_pose),
           world_up_camera_angle_delta_deg=directional_deg(legacy_up,corrected_up)))
    return rows


def main():
    a=args_parser()
    if a.depth_min>=a.depth_max or a.frame_stride<=0:
        raise ValueError("Invalid depth range or frame stride.")
    root=Path(a.gc6d_root).resolve()
    ids=json.loads(Path(a.filtered_scenes).read_text(encoding="utf-8"))
    if not isinstance(ids,list) or len(ids)!=232 or len(set(map(int,ids)))!=232:
        raise ValueError("Expected 232 filtered GC6D training scenes")
    ids=sorted(map(int,ids))
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    dataset=GraspClutter6DMultiDataset(
        root=str(root),camera=a.camera,split="train",scene_ids=ids,max_ann=a.max_ann,
        load_label=False, use_gt_depth=False, use_fuse_depth=False,
        mask_mode="workspace_depth",remove_outlier=None,
        workspace_pose_mode=a.workspace_pose_mode, workspace_outlier=a.workspace_outlier,
        factor_depth_mode=a.factor_depth_mode,fixed_factor_depth=a.fixed_factor_depth,
        graspness_mode="instance")
    if len(dataset)!=3016:
        raise ValueError(f"Expected 3016 D435 frames, got {len(dataset)}")
    depth_rows=[build_depth_frame(dataset,i,a) for i in range(0,len(dataset),a.frame_stride)]
    save_csv(out/"train_crop_depth_frames.csv",depth_rows)
    vectors=audit_camera_vectors(dataset)
    save_csv(out/"camera_film_vectors.csv",vectors)
    pose_rows=compare_pose_hypotheses(root,dataset,a.pose_scene_limit)
    save_csv(out/"pose_consistency_frames.csv",pose_rows)
    by_hyp={hyp:summary([r["displacement_mm"] for r in pose_rows if r["hypothesis"]==hyp])
            for hyp in ("camera_json_is_c2w","camera_json_is_w2c")}
    weighted={}
    for region in ("foreground","background","all"):
        n=sum(r[f"{region}_joint_valid_pixels"] for r in depth_rows)
        total=sum(r[f"{region}_total_abs_error_mm"] for r in depth_rows)
        weighted[region]=dict(common_pixels=n,weighted_mae_mm=float(total/n) if n else None,
            frame_mean_error_stats=summary([r[f"{region}_mae_mm"] for r in depth_rows if r[f"{region}_mae_mm"] is not None]))
    result=dict(status="complete",selected_frames=len(depth_rows),camera_vectors=len(vectors),
        scene_count=len(ids),training_crop="workspace_depth then foreground bounding box resized nearest to 448x448",
        actual_regression_valid_mask="GT in [depth_min,depth_max], not the depth_prob_weight map",
        depth_supervision_source="project-generated virtual_scenes, not official sensor-free GT",
        observed_sensor_depth_is_not_truth=True,
        pose_consistency_mm=by_hyp,
        camera_vector_disagreement_degrees=summary([r["pose_vector_angle_delta_deg"] for r in vectors]),
        world_up_vector_disagreement_degrees=summary([r["world_up_camera_angle_delta_deg"] for r in vectors]),
        depth_stats=weighted,
        interpretation="Object multi-view pose dispersion tests C2W/W2C but does not prove vertical +Z is gravity. "
          "Original training and checkpoint use legacy metadata; do not silently fix only inference.",
        config=vars(a))
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="complete",outputs=str(out),
         pose_consistency_mm=by_hyp,depth_stats=weighted),indent=2))


if __name__=="__main__":
    main()
