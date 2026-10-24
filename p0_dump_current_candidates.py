#!/usr/bin/env python3
"""Dump native current-CDF candidates and pre-postprocessing sidecars.

Used by P0-B for Stage-1 versus Uniform-PKD candidate/ranking decomposition.
No model behavior is changed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser()
    p.add_argument("--dataset_root",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output_root",required=True)
    p.add_argument("--split",required=True,choices=("train","test_seen","test_similar","test_novel"))
    p.add_argument("--camera",default="realsense")
    p.add_argument("--scene_ids",default="")
    p.add_argument("--sample_interval",type=float,default=0.1)
    p.add_argument("--max_samples",type=int,default=-1)
    p.add_argument("--num_point",type=int,default=20000)
    p.add_argument("--num_workers",type=int,default=2)
    p.add_argument("--min_depth",type=float,default=0.2)
    p.add_argument("--max_depth",type=float,default=1.0)
    p.add_argument("--bin_num",type=int,default=256)
    p.add_argument("--graspness_mode",default="scene")
    p.add_argument("--collision_thresh",type=float,default=0.01)
    p.add_argument("--collision_voxel_size",type=float,default=0.01)
    p.add_argument("--overwrite",type=int,choices=(0,1),default=0)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--device",default="cuda:0")
    return p

ARGS=parser().parse_args(); sys.argv[:]=[sys.argv[0]]

import numpy as np
import torch
from torch.utils.data import DataLoader
from pkd_p0.common import annotation_ids, atomic_json_dump, atomic_npz_dump, scene_ids_for_split, seed_everything
from pkd_p0.repo_adapter import RepoImports, DeterministicSubset, build_current_model, build_dataset, dataset_index_records, decode_current, extract_core_outputs, find_point_cloud, forward_model, postprocess_grasps


def records(dataset: Any) -> List[Tuple[int,int,int]]:
    scenes=set(scene_ids_for_split(ARGS.split,ARGS.scene_ids)); annos=set(annotation_ids(ARGS.sample_interval))
    out=[r for r in dataset_index_records(dataset) if r[1] in scenes and r[2] in annos]
    return out[:ARGS.max_samples] if ARGS.max_samples>0 else out


def main() -> None:
    seed_everything(ARGS.seed); device=torch.device(ARGS.device); repo=RepoImports()
    model,_,contract=build_current_model(repo,checkpoint_path=ARGS.checkpoint,device=device,min_depth=ARGS.min_depth,max_depth=ARGS.max_depth,bin_num=ARGS.bin_num,is_training=False)
    dataset=build_dataset(repo,dataset_root=ARGS.dataset_root,split=ARGS.split,camera=ARGS.camera,num_point=ARGS.num_point,min_depth=ARGS.min_depth,max_depth=ARGS.max_depth,bin_num=ARGS.bin_num,use_fuse_depth=contract.use_fuse_depth,graspness_mode=ARGS.graspness_mode,load_label=False,use_gt_depth=(contract.distill_stage==0))
    selected=records(dataset)
    loader=DataLoader(DeterministicSubset(dataset,[r[0] for r in selected],ARGS.seed),batch_size=1,shuffle=False,num_workers=max(0,ARGS.num_workers),collate_fn=repo.collate_fn,pin_memory=device.type=="cuda",persistent_workers=ARGS.num_workers>0)
    root=Path(ARGS.output_root).expanduser().resolve(); root.mkdir(parents=True,exist_ok=True)
    atomic_json_dump({"experiment":"P0 native current candidate dump","checkpoint":contract.to_dict(),"split":ARGS.split,"sample_interval":ARGS.sample_interval,"collision_thresh":ARGS.collision_thresh,"seed":ARGS.seed},root/"p0_candidate_dump_contract.json")
    started=time.time(); new=0
    for local,batch in enumerate(loader):
        dataset_idx,scene,anno=selected[local]
        path=root/ARGS.split/f"scene_{scene:04d}"/ARGS.camera/f"{anno:04d}.npy"
        side=path.with_suffix(".p0_candidates.npz")
        if path.is_file() and side.is_file() and not ARGS.overwrite:
            continue
        cloud=find_point_cloud(batch,0); sample_seed=ARGS.seed+dataset_idx*1_000_003
        with torch.inference_mode(): output=forward_model(repo,model,contract,batch,device=device,seed=sample_seed)
        decoded=decode_current(repo,output)
        raw=decoded[0].detach().cpu().numpy() if torch.is_tensor(decoded[0]) else np.asarray(decoded[0])
        raw=np.asarray(raw,dtype=np.float32)
        final,counts=postprocess_grasps(repo,raw,point_cloud=cloud,collision_thresh=ARGS.collision_thresh,collision_voxel_size=ARGS.collision_voxel_size,apply_nms=True)
        path.parent.mkdir(parents=True,exist_ok=True); np.save(path,final)
        arrays={"raw_grasps":raw,"final_grasps":final,"scene_id":np.asarray([scene],np.int16),"anno_id":np.asarray([anno],np.int16),"checkpoint_sha256":np.asarray(contract.sha256)}
        for name,tensor in extract_core_outputs(output).items():
            value=tensor.detach().cpu().numpy(); arrays[name]=value.astype(np.float32) if np.issubdtype(value.dtype,np.floating) else value
        atomic_npz_dump(side,compress=False,**arrays)
        new+=1; print(f"[DUMP] {new}/{len(selected)} scene={scene:04d} ann={anno:04d} raw={counts['before']} final={counts['after_nms']} elapsed={(time.time()-started)/60:.1f}m",flush=True)
    atomic_json_dump({"status":"complete","new_frames":new,"elapsed_seconds":time.time()-started},root/f"p0_candidate_dump_complete_{ARGS.split}.json")

if __name__=="__main__": main()
