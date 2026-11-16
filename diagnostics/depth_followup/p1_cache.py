"""P1 cache all 13 GC6D views with frozen 448-depth and RAW camera geometry.

This is a NO-training, sensor-depth-for-offline-geometry diagnostic. Source
RGB, original depth, label, camera C2W and original->448 crop identity are
hashed. Missing virtual_scenes/rendered files are never read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import (check_checkpoint,digest,frozen_depth,gc6d_frame,
                     sha256,write_json)


SCHEMA="gc6d-surface-depth-cache-v1"


def load_scene_ids(path):
    x=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(x,list) or not x or any(type(v)!=int or v<0 for v in x):
        raise ValueError("Need explicit positive/zero integer GC6D scene JSON list")
    if len(set(x))!=len(x):raise ValueError("Duplicate GC6D scene")
    return sorted(x)


def make_dataset(a,scenes):
    from dataset.gc6d_dataset import GraspClutter6DMultiDataset
    ds=GraspClutter6DMultiDataset(
        root=a.gc6d_root,camera="realsense-d435",
        split=a.split,scene_ids=scenes,max_ann=13,
        load_label=False,load_grasp_payload=False,augment=False,
        use_gt_depth=False,use_fuse_depth=False,
        mask_mode="workspace_depth",workspace_pose_mode="json",
        workspace_outlier=.02,workspace_depth_trunc=0.,
        factor_depth_mode="bop",fallback_to_depth_mask_on_empty=False)
    if not a.smoke and (len(scenes)!=78 or len(ds)!=1014):
        raise ValueError("Formal GC6D requires same accepted 78 scenes/1014 frames")
    return ds


def frame_record(frame, file, *, base_sha, code_sha):
    return dict(
        scene=frame["scene"],img_id=frame["img_id"],anno=frame["anno"],
        domain="gc6d",
        prediction_file=str(Path(file).resolve()),
        prediction_sha256=sha256(file),
        base_checkpoint_sha256=base_sha,
        source_sha256=frame["source_sha256"],
        source_paths=frame["source_paths"],
        crop_box=frame["crop_box"].tolist(),
        K=frame["K"].tolist(),
        raw_K=frame["raw_K"].tolist(),
        T_c2w=frame["T_c2w"].tolist(),
        reference_source="sensor_depth_m",
        raw_hw=list(frame["raw_hw"]),
        crop_K_error=frame["crop_K_error"],
        depth_factor_divisor=float(frame["depth_factor_divisor"]),
        producer_code_sha256=code_sha)


def _compare_resume(path,record,frame,checkpoint,code_sha):
    if not path.exists() or record is None:
        return False
    if record.get("prediction_sha256")!=sha256(path):
        raise RuntimeError(f"Existing prediction cache changed: {path}")
    keys=("scene","img_id","anno")
    if any(record.get(k)!=frame[k] for k in keys):
        raise RuntimeError("Existing cache view IDs mismatch")
    if (record.get("base_checkpoint_sha256")!=checkpoint or
            record.get("source_sha256")!=frame["source_sha256"] or
            record.get("producer_code_sha256")!=code_sha):
        raise RuntimeError("Existing cache checkpoint/source/code contract changed")
    if not np.array_equal(np.asarray(record["K"],dtype=np.float32),frame["K"]):
        raise RuntimeError("Existing cache crop K changed")
    if not np.array_equal(np.asarray(record["crop_box"]),frame["crop_box"]):
        raise RuntimeError("Existing crop box changed")
    if not np.array_equal(np.asarray(record["T_c2w"],dtype=np.float32),frame["T_c2w"]):
        raise RuntimeError("Existing camera C2W pose changed")
    with np.load(path,allow_pickle=False) as z:
        p=z["pred_depth"]
    if p.shape!=(448,448) or not np.isfinite(p).all() or np.any(p<=0):
        raise ValueError("Existing cached depth invalid")
    return True


def run(a):
    ids=load_scene_ids(a.gc6d_scenes)
    ds=make_dataset(a,ids)
    from rare_grasp.runtime import Runtime
    rt=Runtime(a.bound_config,a.graspnet_root,"",a.device)
    depth=check_checkpoint(rt)
    selected=list(range(len(ds)))
    if a.limit:
        if not a.smoke or a.limit>=len(selected):
            raise ValueError("Partial frame limits only for separate smoke output")
        selected=selected[:a.limit]
    root=Path(a.output)
    if root.exists() and any(root.iterdir()) and not a.resume:
        raise FileExistsError("Use a new cache directory or explicit identical --resume")
    root.mkdir(parents=True,exist_ok=True)
    mf_path=root/"manifest.json"
    if mf_path.exists() and not a.resume:raise FileExistsError(mf_path)
    code_sha=sha256(__file__)
    old={}
    for p in root.glob("frames/scene_*/img_*.json"):
        r=json.loads(p.read_text(encoding="utf-8"))
        key=(r["scene"],r["img_id"])
        if key in old:raise RuntimeError("Duplicate cached frame sidecar")
        old[key]=r
    records=[]
    for n,idx in enumerate(selected,1):
        frame=gc6d_frame(ds,idx)
        key=(frame["scene"],frame["img_id"])
        file=root/"frames"/f"scene_{frame['scene']:06d}"/f"img_{frame['img_id']:06d}.npz"
        side=file.with_suffix(".json")
        if a.resume and key in old and _compare_resume(
                file,old[key],frame,rt.config["checkpoint_sha256"],code_sha):
            record=old[key]
        else:
            if key in old or file.exists() or side.exists():
                raise RuntimeError("Incomplete/modified cache state; do not overwrite frame")
            pred=frozen_depth(depth,frame,rt.device)
            file.parent.mkdir(parents=True,exist_ok=True)
            with file.open("wb") as stream:
                np.savez_compressed(stream,pred_depth=pred.astype(np.float32))
            record=frame_record(frame,file,base_sha=rt.config["checkpoint_sha256"],
                                code_sha=code_sha)
            write_json(side,record)
        records.append(record)
        if n%26==0 or n==len(selected):
            print(f"[P1 cache] {n}/{len(selected)} scene={key[0]} image={key[1]}",
                  flush=True)
    unique={(r["scene"],r["img_id"]) for r in records}
    if len(unique)!=len(records):
        raise RuntimeError("Repeated GC6D cache frames")
    complete=(not a.smoke and not a.limit and len(records)==1014
              and len({r["scene"] for r in records})==78
              and all(len([r for r in records if r["scene"]==s])==13
                      for s in ids))
    if not complete and not a.smoke:
        raise RuntimeError("Formal cache missing a scene or view")
    manifest=dict(schema=SCHEMA,complete=complete,
        base_checkpoint_sha256=rt.config["checkpoint_sha256"],
        base_config_digest=rt.config_digest,
        bound_config_sha256=sha256(a.bound_config),
        scene_ids_sha256=sha256(a.gc6d_scenes),
        dataset_root=str(Path(a.gc6d_root).resolve()),
        code_sha256=code_sha,common_adapter_sha256=sha256(
            Path(__file__).with_name("common.py")),
        source_dataset_sha256=sha256(Path("dataset/gc6d_dataset.py")),
        camera="realsense-d435",depth_origin="RGB+pose, no sensor depth input",
        offline_reference="raw sensor depth; not rendered GT",
        crop="workspace_depth native, C2W scene_camera.json",
        frames=len(records),scenes=len(ids),records=records,
        notes=["Test scenes have been previously inspected and are exploratory",
               "No model fine-tuning, no post-hoc depth scaling or new AP",
               "Original segmentation and sensor depth used for crop and later visibility checks"])
    manifest["manifest_sha256"]=digest(manifest)
    if mf_path.exists():
        if json.loads(mf_path.read_text(encoding="utf-8"))!=manifest:
            raise RuntimeError("Resume manifest differs from immutable source")
    else:
        write_json(mf_path,manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bound-config",required=True)
    p.add_argument("--graspnet-root",required=True)
    p.add_argument("--gc6d-root",required=True)
    p.add_argument("--gc6d-scenes",required=True)
    p.add_argument("--split",default="test")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--output",required=True)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    run(a)


if __name__=="__main__":
    main()
