#!/usr/bin/env python3
"""P0-D: GC6D official test inference/evaluation for Point RayRoPE / GVAR CVA-CDF.

No training. Strict checkpoint state and saved grasp_design config. Outputs
GC6D-native .npy grasps under six-digit scene/camera/image IDs and evaluates
using GraspClutter6DEval.eval_all (official API). Default collision filter off.

IMPORTANT: the existing GC6D loader uses sensor depth/segmentation to choose
workspace/crop (same historical evaluation protocol). The *neural model*
predicts geometry from RGB, K, camera orientation; end-to-end preprocessing is
NOT sensor-free RGB-only. This distinction is recorded in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gc6d-root",required=True)
    p.add_argument("--checkpoint",default=None)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--camera",default="realsense-d435")
    p.add_argument("--split",default="test",choices=("test",))
    p.add_argument("--pose-depth-mode",default="global_film",
                   choices=("none","global_film","ray_gravity_film"))
    p.add_argument("--num-point",type=int,default=20000)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--eval-workers",type=int,default=12)
    p.add_argument("--sample-fraction",type=float,default=1.0)
    p.add_argument("--max-infer-frames",type=int,default=0,help="Smoke inference only; official AP requires all test frames.")
    p.add_argument("--do-infer",action="store_true")
    p.add_argument("--do-eval",action="store_true")
    p.add_argument("--skip-existing",action="store_true")
    p.add_argument("--collision-thresh",type=float,default=0.,
                   help="0 means no depth-dependent postfilter (recommended).")
    p.add_argument("--collision-voxel-size",type=float,default=.01)
    p.add_argument("--max-width",type=float,default=.14,
                   help="Native GC6D clamp; report as evaluation setting.")
    p.add_argument("--mask-mode",default="workspace_depth",
                   choices=("workspace_depth","depth","bbox_depth","label_depth"))
    p.add_argument("--workspace-pose-mode",default="json",choices=("json","inverse","none"))
    p.add_argument("--workspace-outlier",type=float,default=.02)
    p.add_argument("--factor-depth-mode",default="bop",choices=("bop","camera","fixed"))
    p.add_argument("--fixed-factor-depth",type=float,default=None)
    p.add_argument("--min-depth",type=float,default=.2)
    p.add_argument("--max-depth",type=float,default=1.)
    p.add_argument("--bin-num",type=int,default=256)
    p.add_argument("--seed",type=int,default=0)
    return p.parse_args()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):
            h.update(b)
    return h.hexdigest()


def check_prediction(file):
    x=np.load(file,allow_pickle=False)
    if x.ndim!=2 or x.shape[1]!=17 or not np.isfinite(x).all():
        raise ValueError(f"Expected finite Nx17 grasp dump, got {x.shape}: {file}")
    return int(len(x))


def legacy_config_for_model(args):
    """utils.arguments parses argv on import; use it only with isolated argv.

    Changing cfgs must happen before importing model modules that use the
    same global config object.
    """
    oldargv=sys.argv
    try:
        sys.argv=[oldargv[0]]
        from utils.arguments import cfgs
    finally:
        sys.argv=oldargv
    cfgs.multi_modal=True
    cfgs.use_cdf=True
    cfgs.kview_mode="A1"
    cfgs.use_obs_depth=False
    cfgs.use_gt_depth=False
    cfgs.pose_depth_mode=args.pose_depth_mode
    cfgs.min_depth=args.min_depth
    cfgs.max_depth=args.max_depth
    cfgs.bin_num=args.bin_num
    cfgs.num_point=args.num_point
    return cfgs


def make_gc6d_dataset(a):
    from dataset.gc6d_dataset import GraspClutter6DMultiDataset
    return GraspClutter6DMultiDataset(
        root=a.gc6d_root,camera=a.camera,split="test",
        num_points=a.num_point,load_label=False,augment=False,
        load_gt_supervision=False,
        use_gt_depth=False,use_fuse_depth=False,depth_strides=1,
        min_depth=a.min_depth,max_depth=a.max_depth,bin_num=a.bin_num,
        mask_mode=a.mask_mode,remove_outlier=None,
        workspace_pose_mode=a.workspace_pose_mode,workspace_outlier=a.workspace_outlier,
        factor_depth_mode=a.factor_depth_mode,fixed_factor_depth=a.fixed_factor_depth,
        extend_angle=True,
    )


def select_indices(dataset,a):
    if not 0<a.sample_fraction<=1:
        raise ValueError("--sample-fraction must lie in (0,1]")
    selected=[]
    stride=1 if a.sample_fraction>=1 else max(1,round(1./a.sample_fraction))
    groups={}
    for i,(sid,img_id,ann) in enumerate(dataset.samples):
        groups.setdefault(int(sid),[]).append(i)
    for sid in sorted(groups):
        ordered=sorted(groups[sid],key=lambda i:dataset.samples[i][2])
        selected.extend(ordered[::stride])
    if a.max_infer_frames:
        selected=selected[:a.max_infer_frames]
    return selected


def identities(dataset,selected,output):
    return [(f"{int(dataset.samples[i][0]):06d}",
             f"{int(dataset.samples[i][1]):06d}",
             output / f"{int(dataset.samples[i][0]):06d}" / dataset.camera /
             f"{int(dataset.samples[i][1]):06d}.npy") for i in selected]


def make_model(a):
    import torch
    from tools.mixed_p0_model_checkpoint import (
        read_checkpoint, design_init_options, load_strict_model,
    )
    from models.economicgrasp_bip3d import economicgrasp_dpt
    ck,state=read_checkpoint(a.checkpoint)
    kwargs=design_init_options(ck)
    model=economicgrasp_dpt(
        **kwargs,min_depth=a.min_depth,max_depth=a.max_depth,bin_num=a.bin_num,
        is_training=False,use_obs_depth=False,use_cdf=True,
        pose_depth_mode=a.pose_depth_mode,
    )
    load_strict_model(model,ck,state)
    model.cuda().eval()
    return model, ck["grasp_design"]


def run_inference(a,dataset,selected,output,manifest):
    import torch
    from torch.utils.data import DataLoader,Subset
    from dataset.graspnet_dataset import collate_fn
    from models.economicgrasp_bip3d import pred_decode_center_view_angle
    from graspnetAPI import GraspGroup
    from utils.collision_detector import ModelFreeCollisionDetectorTorch
    if not torch.cuda.is_available():
        raise RuntimeError("P0-D model inference requires one available CUDA GPU.")
    model,design=make_model(a)
    manifest["grasp_design"]=design
    paths=identities(dataset,selected,output)
    loader=DataLoader(Subset(dataset,selected),batch_size=a.batch_size,shuffle=False,
                      num_workers=a.workers,collate_fn=collate_fn,
                      persistent_workers=(a.workers>0),pin_memory=False)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    processed=0
    for bi,batch in enumerate(loader):
        for key,value in batch.items():
            if isinstance(value,(tuple,list)):
                raise TypeError(f"Unexpected list input {key}; dataset must have load_label=False")
            if torch.is_tensor(value):
                batch[key]=value.cuda(non_blocking=False)
        # Existing GC6D inference loader now supplies the exact training
        # camera_pose_vec / camera_gravity_vec convention.
        for key in ("img","K","camera_pose_vec","camera_gravity_vec"):
            if key not in batch:
                raise KeyError(f"Missing input {key} in GC6D inference batch")
        batch["cva_export_angle_feature"]=False
        with torch.inference_mode():
            endpoints=model(batch)
            predicted=pred_decode_center_view_angle(endpoints,use_cdf=True)
        for i,pred in enumerate(predicted):
            loc=bi*a.batch_size+i
            if loc>=len(paths):
                raise RuntimeError("Dataloader/prediction size mismatch")
            _,_,file=paths[loc]
            if a.skip_existing and file.exists():
                check_prediction(file)
                continue
            group=GraspGroup(pred.detach().cpu().numpy())
            if len(group):
                mat=group.grasp_group_array
                mat[:,1]=np.minimum(mat[:,1],float(a.max_width))
                group.grasp_group_array=mat
            if a.collision_thresh>0:
                # Explicitly a sensor-cloud privilege, only for separately
                # labeled legacy-protocol comparisons, never RGB-only claim.
                cloud,_=dataset.get_data(selected[loc],return_raw_cloud=True)
                det=ModelFreeCollisionDetectorTorch(cloud.reshape(-1,3),
                          voxel_size=a.collision_voxel_size)
                coll=det.detect(group,approach_dist=.05,
                                collision_thresh=a.collision_thresh)
                group=group[~coll.detach().cpu().numpy()]
            file.parent.mkdir(parents=True,exist_ok=True)
            group.save_npy(str(file))
            check_prediction(file)
            processed+=1
        if bi%20==0:
            print(f"[GC6D-CVA] batches={bi+1}/{len(loader)} newly_saved={processed}",flush=True)
    expected=len(selected)
    counts=[check_prediction(f) for _,_,f in paths]
    manifest["dump_stats"]=dict(expected=expected,validated=len(counts),
                                empty_prediction_frames=sum(n==0 for n in counts),
                                saved_new=processed,all_finite_nx17=True)


def evaluate_official(a,dataset,output,manifest):
    from graspclutter6dAPI import GraspClutter6DEval
    if a.sample_fraction!=1.0 or a.max_infer_frames:
        raise ValueError("Official GC6D eval requires all frames: --sample-fraction 1 --max-infer-frames 0")
    selected=list(range(len(dataset)))
    files=identities(dataset,selected,output)
    counts=[check_prediction(path) for _,_,path in files]
    evaluator=GraspClutter6DEval(root=str(a.gc6d_root),camera=dataset.camera,split="test")
    res,ap=evaluator.eval_all(str(output),proc=a.eval_workers)
    data=np.asarray(res)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite official GC6D evaluation array")
    np.save(output/"gc6d_official_raw_accuracy.npy",data)
    result=np.asarray(ap,dtype=float).reshape(-1)
    if result.size<3 or not np.isfinite(result).all():
        raise ValueError(f"Unexpected GraspClutter6DEval.eval_all AP returned: {ap}")
    manifest["official_eval"]=dict(
        api="GraspClutter6DEval.eval_all", split="test",camera=dataset.camera,
        ap_returned_raw=result.tolist(),ap_raw_units="as returned by installed official API",
        ap_label_order_assumed=["AP","AP0.4","AP0.8"],
        raw_accuracy_shape=list(data.shape),num_evaluated_frames=len(files),
        empty_prediction_frames=sum(n==0 for n in counts),
        raw_accuracy_file="gc6d_official_raw_accuracy.npy")


def main():
    a=parse_args()
    if not a.do_infer and not a.do_eval:
        a.do_infer=a.do_eval=True
    if a.do_infer and not a.checkpoint:
        raise ValueError("--checkpoint mandatory for --do-infer.")
    if a.batch_size<1 or a.workers<0 or a.eval_workers<1:
        raise ValueError("Invalid batch/workers")
    if a.collision_thresh<0 or a.max_width<=0:
        raise ValueError("Invalid collision threshold or width")
    a.gc6d_root=str(Path(a.gc6d_root).expanduser().resolve())
    a.camera={"realsense":"realsense-d435","kinect":"azure-kinect"}.get(a.camera,a.camera)
    output=Path(a.output_dir).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    legacy_config_for_model(a)
    dataset=make_gc6d_dataset(a)
    selected=select_indices(dataset,a)
    if a.do_eval and (len(selected)!=len(dataset) or a.sample_fraction!=1.0 or a.max_infer_frames):
        raise ValueError("Never evaluate partial GC6D dump with eval_all(); use complete official test set.")
    args=dict(vars(a));args["gc6d_root"]=str(a.gc6d_root)
    manifest=dict(status="prepared",run_args=args,selected_count=len(selected),
        total_test_frames=len(dataset), camera=dataset.camera,
        checkpoint_sha256=file_hash(a.checkpoint) if a.checkpoint else None,
        input_modality="RGB+known K/pose for neural network, predicted metric geometry",
        preprocessing_privilege="sensor depth and segmentation used by existing GC6D workspace/crop loader",
        gt_virtual_depth_loaded=False,
        model_free_collision_filter="off" if a.collision_thresh==0 else "sensor_depth_on",
        pose_frame_interpretation="same legacy GC6D training loader global_film; verify using P0-C",
        split="test")
    manifest_file=output/"gc6d_p0_d_manifest.json"
    if manifest_file.is_file():
        previous=json.loads(manifest_file.read_text(encoding="utf-8"))
        if a.do_infer:
            # Never overwrite dumps from another checkpoint, even when
            # --skip-existing is false: move to a new output directory.
            if previous.get("checkpoint_sha256")!=manifest["checkpoint_sha256"]:
                raise ValueError("Output directory contains a different checkpoint manifest.")
        elif a.do_eval:
            if not previous.get("checkpoint_sha256"):
                raise ValueError("Evaluation-only requires prior inference manifest with checkpoint hash.")
            manifest=previous
            manifest["do_eval_only"]=True
    elif a.do_eval and not a.do_infer:
        raise FileNotFoundError("Evaluation-only requires existing gc6d_p0_d_manifest.json")
    # Persist the manifest before long inference, so interrupted runs can be
    # resumed with --skip-existing without changing checkpoint identity.
    manifest_file.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    t=time.time()
    if a.do_infer:
        run_inference(a,dataset,selected,output,manifest)
    if a.do_eval:
        evaluate_official(a,dataset,output,manifest)
    manifest["status"]="complete"
    manifest["runtime_seconds"]=round(time.time()-t,2)
    manifest_file.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="complete",manifest=str(manifest_file),
                          official_eval=manifest.get("official_eval")),indent=2))


if __name__=="__main__":
    main()
