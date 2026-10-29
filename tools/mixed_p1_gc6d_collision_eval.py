#!/usr/bin/env python3
"""P1-0 and collision-on GC6D official AP, no training or checkpoint modification.

Supports a historical *two-source* Point checkpoint and the October 27
*three-source* Point / Point+GVAR checkpoints with the appropriate historical
source tree for each. Do NOT import the current main economicgrasp_dpt as a
substitute: historical checkpoints use protocol/model and a pinned code SHA.

Produces a fresh collision-on dump tree and native official GraspClutter6DEval
AP. An optional collision-off tree is generated from the SAME neural forward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
# Legacy graspnetAPI/graspclutter6dAPI releases may still reference np.float.
if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]
import torch
from torch.utils.data import DataLoader

from mixed_p1_historical import (
    activate_historical, gc6d_test_dataset, load_model, sha256,
    DeterministicIndexedDataset,
)


MU = (.2,.4,.6,.8,1.,1.2)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-root", required=True,
                   help="Source worktree that created THIS checkpoint, must match its fingerprint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--gc6d-root", required=True)
    p.add_argument("--output-root", required=True,
                   help="Fresh unique result root per checkpoint; never use existing P0 off directory.")
    p.add_argument("--camera", default="realsense-d435")
    p.add_argument("--mode", choices=("infer","eval","both"), default="both")
    p.add_argument("--collision", choices=("on","off","both"), default="on")
    p.add_argument("--collision-thresh",type=float,default=.01)
    p.add_argument("--collision-voxel-size",type=float,default=.01)
    p.add_argument("--collision-approach-dist",type=float,default=.05)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--num-workers",type=int,default=3)
    p.add_argument("--eval-workers",type=int,default=12)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--max-frames",type=int,default=0,
                   help="0 means all 3055 frames; nonzero only for inference smoke (never official AP).")
    p.add_argument("--resume",action="store_true",
                   help="Reuse matching validated dump files; do not overwrite other checkpoints.")
    p.add_argument("--accept-preprocessing-privilege",action="store_true",
                   help="Acknowledge sensor depth+segmentation workspace/crop, and sensor depth collision filtering.")
    return p.parse_args(argv)


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp."+str(os.getpid()))
    try:
        tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_npy(path,arr):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp."+str(os.getpid()))
    try:
        with tmp.open("wb") as f:np.save(f,arr,allow_pickle=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def validate_grasps(file):
    a=np.load(file,allow_pickle=False)
    if a.ndim!=2 or a.shape[1]!=17 or not np.isfinite(a).all():
        raise ValueError(f"Expected finite Nx17 grasp predictions at {file}; got {a.shape}")
    return a


def expected_paths(dataset, output, mode, selected):
    paths=[]
    for idx in selected:
        sid,img,ann=dataset.samples[idx]
        paths.append((int(sid),int(img),int(ann),
                      Path(output)/f"collision_{mode}"/f"{int(sid):06d}"/
                      dataset.camera/f"{int(img):06d}.npy"))
    if len({(x[0],x[1]) for x in paths})!=len(paths):
        raise RuntimeError("GC6D dataset has repeated scene/image IDs")
    return paths


def full_identity_audit(paths, expected=3055):
    if len(paths)!=expected:
        raise RuntimeError(f"Expected {expected} GC6D image identities, got {len(paths)}")
    missing=[str(row[3]) for row in paths if not row[3].is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} GC6D dumps; first: {missing[:8]}"
        )
    counts=[len(validate_grasps(row[3])) for row in paths]
    digest=hashlib.sha256(
        "".join(f"{sid:06d}/{img:06d}\n" for sid,img,ann,_ in paths).encode()
    ).hexdigest()
    return dict(frames=len(paths),total_saved_grasps=int(sum(counts)),
                empty_frames=int(sum(x==0 for x in counts)),
                min_grasps=int(min(counts)),max_grasps=int(max(counts)),
                identity_sha256=digest,
                finite_nx17_verified=True)


def make_manifest(a,model_meta,dataset):
    modes=["on","off"] if a.collision=="both" else [a.collision]
    source=Path(a.historical_root).expanduser().resolve()
    cp=Path(a.checkpoint).expanduser().resolve()
    return dict(
        checkpoint=str(cp),checkpoint_sha256=sha256(cp),
        historical_model_root=str(source),
        historical_source_sha256=model_meta["source_sha256"],
        historical_dav2_sha256=model_meta["dav2_sha256"],
        model_variant=model_meta["variant"],
        completed_epochs=model_meta["completed_epochs"],
        gc6d_root=str(Path(a.gc6d_root).resolve()),
        gc6d_camera=dataset.camera,official_split="test",
        official_scenes=len(dataset.scene_ids),official_frames=len(dataset),
        input="RGB + K + camera orientation, predicted metric depth",
        preprocessing="GC6D sensor depth and segmentation used to crop/workspace",
        model_free_collision="sensor depth; scene/model collision check distinct from official evaluator",
        filter_modes=modes,
        collision_params=dict(threshold=a.collision_thresh,
            voxel_size_m=a.collision_voxel_size,approach_dist_m=a.collision_approach_dist),
        inference_args=dict(batch_size=a.batch_size,num_workers=a.num_workers,seed=a.seed),
        official_eval_mu=list(MU),
        dataset_loader_sha256=sha256(Path(__file__).resolve().parents[1]/"dataset"/"gc6d_dataset.py"),
    )


def ensure_manifest(directory,payload,resume):
    file=Path(directory)/"inference_protocol.json"
    if file.exists():
        old=json.loads(file.read_text(encoding="utf-8"))
        if old!=payload:
            raise RuntimeError(f"Existing directory has an incompatible checkpoint/data/filter protocol: {file}")
        if not resume:
            raise FileExistsError(f"Output already exists, set --resume to reuse: {file}")
    else:
        if any(Path(directory).iterdir()):
            raise FileExistsError(f"Output root must be empty or have the exact matching manifest: {directory}")
        atomic_json(file,payload)
    return file


def infer(a,dataset,selected,manifest,output,model):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for historical RGB grasp inference and collision filter")
    from moge_rayrope.runtime import move_batch
    from dataset.graspnet_dataset import collate_fn
    from graspnetAPI import GraspGroup
    from utils.collision_detector import ModelFreeCollisionDetectorTorch
    from models.economicgrasp_bip3d import pred_decode_center_view_angle

    modes=manifest["filter_modes"]
    paths={mode:expected_paths(dataset,output,mode,selected) for mode in modes}
    ds=DeterministicIndexedDataset(dataset,selected,salt=34153)
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,
                      num_workers=a.num_workers,collate_fn=collate_fn,
                      persistent_workers=bool(a.num_workers))
    model.eval()
    stats={mode:dict(before=0,after=0,empty_after=0) for mode in modes}
    start=time.monotonic()
    for bi,raw in enumerate(loader):
        n=len(raw["img"])
        start_idx=bi*a.batch_size
        # No overwrite of existing samples unless exact manifest and --resume.
        if a.resume and all(
            paths[mode][start_idx+i][3].is_file()
            for mode in modes for i in range(n)
        ):
            for mode in modes:
                for i in range(n):
                    arr=validate_grasps(paths[mode][start_idx+i][3])
                    stats[mode]["after"]+=len(arr)
                    stats[mode]["empty_after"]+=int(len(arr)==0)
            continue
        # This uses the existing historical move_batch(..., labels=False), which
        # strips observed depth and labels before forward. Do not substitute
        # current-main model/dataset contracts.
        np.random.seed(a.seed+start_idx)
        torch.manual_seed(a.seed+start_idx)
        with torch.no_grad():
            inputs=move_batch(raw,"cuda:0",labels=False)
            if "sensor_depth_m" in inputs or "gt_depth_m" in inputs:
                raise RuntimeError("Privileged sensor/GT depth entered model")
            for key in ("img","K","camera_pose_vec"):
                if key not in inputs:
                    raise KeyError(f"Missing neural input {key}")
            ep=model(inputs)
            predictions=pred_decode_center_view_angle(ep,use_cdf=True)
        if len(predictions)!=n:
            raise RuntimeError(f"Model emitted {len(predictions)} predictions for batch {n}")
        for j,pred in enumerate(predictions):
            actual_idx=start_idx+j
            arr=pred.detach().float().cpu().numpy()
            if arr.ndim!=2 or arr.shape[1]!=17 or not np.isfinite(arr).all():
                raise ValueError(f"Nonfinite or non-Nx17 network output index={actual_idx}")
            output_arrays={"off":arr}
            if "on" in modes:
                cloud,_=dataset.get_data(selected[actual_idx],return_raw_cloud=True)
                det=ModelFreeCollisionDetectorTorch(
                    np.asarray(cloud,np.float32).reshape(-1,3),
                    voxel_size=a.collision_voxel_size)
                gg=GraspGroup(arr)
                keep=~det.detect(
                    gg,approach_dist=a.collision_approach_dist,
                    collision_thresh=a.collision_thresh
                ).detach().cpu().numpy()
                if len(keep)!=len(arr):
                    raise RuntimeError("Collision filter mask length mismatch")
                output_arrays["on"]=arr[keep]
            for mode in modes:
                path=paths[mode][actual_idx][3]
                if a.resume and path.is_file():
                    old=validate_grasps(path)
                    if old.shape != output_arrays[mode].shape or not np.allclose(
                        old,output_arrays[mode],atol=1e-6,rtol=1e-6):
                        raise RuntimeError(f"Resuming would overwrite changed inference: {path}")
                else:
                    atomic_npy(path,output_arrays[mode])
                    validate_grasps(path)
                stats[mode]["before"]+=len(arr)
                stats[mode]["after"]+=len(output_arrays[mode])
                stats[mode]["empty_after"]+=int(len(output_arrays[mode])==0)
        if bi % 20 == 0:
            print(f"[P1-GC6D] {bi+1}/{len(loader)} batches, {start_idx+n}/{len(ds)} frames",flush=True)
    summary=dict(model=manifest["model_variant"],selected_frames=len(selected),
                 elapsed_sec=time.monotonic()-start,filter_counts=stats)
    atomic_json(output/"inference_complete.json",summary)
    return summary


def evaluate(a,dataset,output,manifest):
    if a.max_frames:
        raise ValueError("GC6D official AP cannot use --max-frames smoke subset")
    if len(dataset)!=3055:
        raise RuntimeError("GC6D official test must be 235×13 complete")
    from graspclutter6dAPI import GraspClutter6DEval
    result={}
    for mode in manifest["filter_modes"]:
        dir=Path(output)/f"collision_{mode}"
        paths=expected_paths(dataset,output,mode,range(len(dataset)))
        integrity=full_identity_audit(paths,expected=3055)
        # Native GC6D eval_scene returns zero accuracy for an absent dump.
        # Empty Nx17 files after filtering can fail during API NMS. Preserve
        # every prediction file and expose only non-empty grasps via an
        # isolated symlink view; empty predictions get official zero accuracy.
        from contextlib import nullcontext
        empty_paths=[p for _,_,_,p in paths if len(validate_grasps(p))==0]
        staged=tempfile.TemporaryDirectory(prefix="gc6d_eval_empty_",dir=str(output)) if empty_paths else nullcontext(None)
        with staged as tmp:
            official_dump=dir
            if tmp is not None:
                official_dump=Path(tmp)
                for sid,img,ann,p in paths:
                    if len(validate_grasps(p))==0:
                        continue
                    link=official_dump/f"{sid:06d}"/dataset.camera/f"{img:06d}.npy"
                    link.parent.mkdir(parents=True,exist_ok=True)
                    link.symlink_to(p.resolve())
            evaluator=GraspClutter6DEval(root=str(Path(a.gc6d_root).resolve()),
                                        camera=dataset.camera,split="test")
            res,ap=evaluator.eval_all(str(official_dump),proc=a.eval_workers)
        accuracy=np.asarray(res,np.float32)
        if accuracy.shape!=(235,13,50,6) or not np.isfinite(accuracy).all():
            raise ValueError(f"Official GC6D accuracy invalid shape/range: {accuracy.shape}")
        if accuracy.min()<0 or accuracy.max()>1:
            raise ValueError("GC6D official accuracy not in [0,1]")
        returned=np.asarray(ap,np.float64).reshape(-1)
        recalculated=np.array([
            accuracy.mean(),accuracy[...,1].mean(),accuracy[...,3].mean()
        ],np.float64)
        if returned.size<3 or not np.allclose(returned[:3],recalculated,atol=2e-6):
            raise ValueError(f"Official AP return vs full array mismatch: {returned}, {recalculated}")
        atomic_npy(dir/"official_accuracy.npy",accuracy)
        row=dict(mode=mode,ap_percent=float(100*recalculated[0]),
                 ap04_percent=float(100*recalculated[1]),
                 ap08_percent=float(100*recalculated[2]),
                 by_mu_percent=[float(v*100) for v in accuracy.mean(axis=(0,1,2))],
                 official_empty_prediction_zero_fallback_frames=len(empty_paths),
                 raw_accuracy_sha256=sha256(dir/"official_accuracy.npy"),
                 **integrity)
        atomic_json(dir/"official_summary.json",row)
        result[mode]=row
    atomic_json(output/"evaluation_complete.json",result)
    return result


def main(argv=None):
    a=parse_args(argv)
    if not a.accept_preprocessing_privilege:
        raise ValueError(
            "Acknowledge RGB neural geometry + sensor-depth workspace/crop/collision via "
            "--accept-preprocessing-privilege"
        )
    if a.batch_size<1 or a.num_workers<0 or a.eval_workers<1 or a.max_frames<0:
        raise ValueError("Invalid batch/workers/max_frames")
    if not (0<a.collision_thresh<=.1 and 0<a.collision_voxel_size<=.1 and 0<a.collision_approach_dist<=.2):
        raise ValueError("Invalid positive collision parameters")
    activate_historical(a.historical_root)
    model,ck,model_meta=load_model(a.historical_root,a.checkpoint,"cuda:0")
    if model_meta["variant"] not in ("rayrope_point","rayrope_point_gvar"):
        raise ValueError("Only historical Point/GVAR checkpoints are supported")
    ds=gc6d_test_dataset(a.gc6d_root,camera=a.camera)
    if a.max_frames>len(ds):
        raise ValueError(f"--max-frames={a.max_frames} exceeds GC6D test size={len(ds)}")
    selected=list(range(len(ds))) if not a.max_frames else list(range(a.max_frames))
    output=Path(a.output_root).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    manifest=make_manifest(a,model_meta,ds)
    ensure_manifest(output,manifest,resume=a.resume)
    result={}
    if a.mode in ("infer","both"):
        result["inference"]=infer(a,ds,selected,manifest,output,model)
    if a.mode in ("eval","both"):
        result["evaluation"]=evaluate(a,ds,output,manifest)
    atomic_json(output/"run_complete.json",dict(manifest=manifest,results=result))
    print(json.dumps(dict(status="complete",output=str(output),
             model=model_meta["variant"],results=result.get("evaluation")),indent=2))


if __name__=="__main__":
    main()
