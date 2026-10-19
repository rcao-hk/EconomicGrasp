#!/usr/bin/env python3
"""Full 100% Original GraspNet RealSense training for legacy/CDF point baselines.

Only the final point grasp head and its label/loss/decode contract differ.
All models: RGB + global_film DAV2/DPT metric depth, detached backprojected
point cloud, original EconomicGrasp3D Minkowski backbone/center/view/grouping.

Default: train all 100 scenes x 256 frames, Original Seen 10% validation.
Use --max_train_batches/--max_val_batches ONLY for preflight smoke runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def _extra_options():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument("--point_head", choices=("legacy","cdf"), required=True)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--shared_depth_checkpoint", default="")
    x,rest=p.parse_known_args()
    sys.argv=[sys.argv[0],*rest]
    if not (0.0<x.val_fraction<=1.0):
        raise ValueError("val_fraction must be (0,1]")
    if min(x.max_train_batches,x.max_val_batches)<0:
        raise ValueError("max batches must be >=0")
    return x


FULL=_extra_options()

import numpy as np
import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

import train_ddp as dist_utils
from dataset.graspnet_dataset import GraspNetMultiDataset,collate_fn
from dataset.cdf_label_adapter import CVAExtendedLabelAdapter
from models.economicgrasp_depth import economicgrasp_depth_baseline
from models.loss_economicgrasp_depth import get_loss as legacy_loss
from models.loss_point_depth_cdf import get_point_cdf_loss
from utils.arguments import cfgs
from utils.point_depth_mixed_contract import scene_stride_indices,scene_frame_signature

CONTRACT_VERSION=1
PARENT_BRANCH="exp/point-depth-pose-aware-mixed"


def source_sha():
    try:
        return subprocess.check_output(
            ("git","rev-parse","HEAD"),text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path):
    digest=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def make_dataset(split):
    cdf=FULL.point_head=="cdf"
    base=GraspNetMultiDataset(
        cfgs.dataset_root,camera=cfgs.camera,split=split,
        num_points=cfgs.num_point,voxel_size=cfgs.voxel_size,
        remove_outlier=True,augment=False,load_label=True,
        use_gt_depth=False,use_fuse_depth=bool(cfgs.use_fuse_depth),
        graspness_mode="scene",min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,bin_num=cfgs.bin_num,
        depth_strides=1,extend_angle=cdf,load_grasp_payload=not cdf,
    )
    if not cdf:
        return base
    return CVAExtendedLabelAdapter(
        base,dataset_root=cfgs.dataset_root,
        use_cdf=True,
        label_folder="economic_grasp_label_300views_extend_angle_cdf_depth",
        num_angle=cfgs.num_angle,num_depth=cfgs.num_depth,
    )


def move_to_gpu(batch,device,head):
    """Preserve CPU object caches for CDF's sparse per-query matcher.

    DDP(device_ids=None) is REQUIRED or DDP will recursively move these lists.
    Legacy label matching expects GPU tensors; only legacy list payloads move.
    """
    for key,value in list(batch.items()):
        if isinstance(value,(list,tuple)):
            if head=="cdf":
                # CPU-resident source lists; matcher transfers only selected rows
                for sample in value:
                    for tensor in sample:
                        if torch.is_tensor(tensor) and tensor.device.type!="cpu":
                            raise RuntimeError(f"{key}: CDF object labels must stay on CPU")
                continue
            batch[key]=[
                [item.to(device,non_blocking=False) if torch.is_tensor(item) else item
                 for item in sample] for sample in value
            ]
        elif torch.is_tensor(value):
            batch[key]=value.to(device,non_blocking=False)
    return batch


class Trainer:
    def __init__(self):
        if cfgs.camera!="realsense":
            raise ValueError("Full-data controlled experiment requires RealSense")
        if cfgs.pose_depth_mode!="global_film":
            raise ValueError("Requires the same global_film pose-aware DPT")
        if cfgs.use_gt_depth or cfgs.use_obs_depth or cfgs.use_depth_comp:
            raise ValueError("GT/observed depth is not permitted as grasp geometry")
        if not cfgs.use_fuse_depth:
            raise ValueError("Full-data primary contract: rendered foreground + TSDF background")
        if cfgs.num_angle!=12 or cfgs.num_depth!=4 or cfgs.num_view!=300:
            raise ValueError("Expected 300 views x12 angles x4 insertion depths")
        if cfgs.min_depth!=0.2 or cfgs.max_depth!=1.0:
            raise ValueError("Metric depth target must use 0.2..1.0 metres")
        if cfgs.pin_memory:
            raise ValueError("Variable-length grasp payload must not use pin_memory")
        if cfgs.checkpoint_path and not cfgs.resume:
            raise ValueError("Fresh initialization required unless explicitly --resume")
        if cfgs.resume and not cfgs.checkpoint_path:
            raise ValueError("--resume requires --checkpoint_path")
        if FULL.shared_depth_checkpoint and cfgs.resume:
            raise ValueError("Shared-depth initialization and resume are mutually exclusive")

        self.distributed,self.rank,self.local_rank,self.world_size,self.device=(
            dist_utils.setup_distributed()
        )
        self.main=self.rank==0
        dist_utils.seed_everything(cfgs.seed,self.rank)
        self.directory=Path(cfgs.log_dir)
        self.directory.mkdir(parents=True,exist_ok=True)

        train=make_dataset("train")
        seen=make_dataset("test_seen")
        val_idx=scene_stride_indices(seen,FULL.val_fraction)
        self.train_dataset=train
        self.val_dataset=Subset(seen,val_idx)
        if len(train)!=25600:
            raise RuntimeError(f"Expected all 25,600 Original train frames; got {len(train)}")
        if FULL.val_fraction==0.1 and len(val_idx)!=780:
            raise RuntimeError(f"Expected Seen validation 780, got {len(val_idx)}")

        self.protocol={
            "contract_version":CONTRACT_VERSION,
            "dataset":"GraspNet-1Billion Original RealSense",
            "training_fraction":1.0,
            "train_frame_count":len(train),
            "train_scene_count":100,
            "validation_split":"test_seen",
            "validation_fraction":FULL.val_fraction,
            "validation_frame_count":len(val_idx),
            "training_scene_frame_sha256":scene_frame_signature(train,list(range(len(train)))),
            "validation_scene_frame_sha256":scene_frame_signature(seen,val_idx),
            "head":FULL.point_head,
            "point_backbone":"original EconomicGrasp3D Minkowski TDUnet; original point/view/cylinder grouping",
            "pose_depth_mode":"global_film",
            "rgb_encoder":"frozen DepthAnythingV2 ViT-B",
            "depth_target":"rendered foreground plus TSDF background",
            "geometry_for_grasp":"detached DPT-predicted metric depth, never observed or GT",
            "grasp_supervision":(
                "standard economic_grasp_label_300views + original CE score/depth/angle & width"
                if FULL.point_head=="legacy" else
                "extended-angle CDF bins + depth-wise width with same CVA unbalanced BCE"
            ),
            "cdf_thresholds":[0.2,0.4,0.6,0.8,1.0,1.2] if FULL.point_head=="cdf" else None,
            "cdf_label_folder":(
                "economic_grasp_label_300views_extend_angle_cdf_depth"
                if FULL.point_head=="cdf" else None
            ),
            "width_cache_conversion":"mm /1000 once; regression target m*10; decode 1.2*(m*10)/10",
            "seed":int(cfgs.seed),"epochs":int(cfgs.max_epoch),
            "batch_size_per_rank":int(cfgs.batch_size),
            "world_size":self.world_size,"global_batch_size":cfgs.batch_size*self.world_size,
            "nominal_samples_per_epoch":len(train),
            "distributed_sampler_padded_samples_per_epoch":math.ceil(len(train)/self.world_size)*self.world_size,
            "optimizer":"AdamW",
            "learning_rate":float(cfgs.learning_rate),
            "weight_decay":float(cfgs.weight_decay),
            "depth_weight_decay":float(cfgs.depth_weight_decay),
            "checkpoint_for_official_ap":"checkpoint_epoch_019.tar, predeclared e19",
            "max_train_batches":FULL.max_train_batches,
            "max_val_batches":FULL.max_val_batches,
            "shared_depth_checkpoint":FULL.shared_depth_checkpoint,
            "git_source_sha":source_sha(),
            "parent_branch":PARENT_BRANCH,
        }

        self.train_sampler=(
            DistributedSampler(train,num_replicas=self.world_size,
                               rank=self.rank,shuffle=True,seed=cfgs.seed)
            if self.distributed else None
        )
        self.val_sampler=(
            DistributedSampler(self.val_dataset,num_replicas=self.world_size,
                               rank=self.rank,shuffle=False)
            if self.distributed else None
        )

        def loader(dataset,sampler,workers,training):
            kw={"prefetch_factor":1} if workers>0 else {}
            return DataLoader(
                dataset,batch_size=cfgs.batch_size,
                sampler=sampler,shuffle=(sampler is None and training),
                num_workers=workers,collate_fn=collate_fn,
                worker_init_fn=dist_utils.my_worker_init_fn,
                persistent_workers=(workers>0 and training),
                drop_last=False,pin_memory=False,**kw
            )

        self.train_loader=loader(
            train,self.train_sampler,max(0,cfgs.num_workers),True
        )
        self.val_loader=loader(
            self.val_dataset,self.val_sampler,max(0,cfgs.eval_num_workers),False
        )

        model=economicgrasp_depth_baseline(
            num_points=cfgs.num_point,voxel_size=cfgs.voxel_size,
            min_depth=cfgs.min_depth,max_depth=cfgs.max_depth,
            is_training=True,use_obs_depth=False,
            pose_depth_mode="global_film",
            point_use_cdf=FULL.point_head=="cdf",
            vis_dir=None,
        ).to(self.device)

        self.shared_depth=bool(FULL.shared_depth_checkpoint)
        if self.shared_depth:
            p=Path(FULL.shared_depth_checkpoint)
            if not p.is_file():
                raise FileNotFoundError(p)
            ck=torch.load(p,map_location="cpu",weights_only=False)
            if ck.get("pose_depth_mode")!="global_film":
                raise ValueError("Shared depth checkpoint must declare global_film")
            state=ck.get("model_state_dict")
            if not isinstance(state,dict):
                raise ValueError("Shared-depth model_state_dict required")
            depth_state={key[10:]:value for key,value in state.items()
                         if key.startswith("depth_net.")}
            model.depth_net.load_state_dict(depth_state,strict=True)
            model.depth_net.requires_grad_(False)
            model.depth_net.eval()
            self.protocol["shared_depth_checkpoint_sha256"]=sha256_file(p)

        self.module=model
        # CPU CDF labels must stay on CPU during DDP _to_kwargs; device_ids=None.
        self.net=DDP(
            model,device_ids=None,broadcast_buffers=False,
            find_unused_parameters=False,gradient_as_bucket_view=True
        ) if self.distributed else model
        depth=[p for p in model.depth_net.parameters() if p.requires_grad]
        d_ids={id(p) for p in depth}
        grasp=[p for p in model.parameters() if p.requires_grad and id(p) not in d_ids]
        if not grasp or (not self.shared_depth and not depth):
            raise RuntimeError("Expected trainable grasp head and depth decoder+FiLM")
        groups=[{"params":grasp,"weight_decay":cfgs.weight_decay}]
        if depth:
            groups.append({"params":depth,"weight_decay":cfgs.depth_weight_decay})
        self.optimizer=optim.AdamW(groups,lr=cfgs.learning_rate)
        self.loss_fn=get_point_cdf_loss if FULL.point_head=="cdf" else legacy_loss
        self.start_epoch=0
        if cfgs.resume:
            ck=torch.load(cfgs.checkpoint_path,map_location="cpu",weights_only=False)
            prior=ck.get("full100_protocol",{})
            for key in ("head","training_scene_frame_sha256","validation_scene_frame_sha256",
                        "shared_depth_checkpoint_sha256"):
                if prior.get(key)!=self.protocol.get(key):
                    raise RuntimeError(f"Resume mismatch in {key}")
            model.load_state_dict(ck["model_state_dict"],strict=True)
            self.optimizer.load_state_dict(ck["optimizer_state_dict"])
            self.start_epoch=int(ck["epoch"])
        if self.main:
            manifest=self.directory/"point_depth_full100_protocol.json"
            if manifest.exists() and not cfgs.resume:
                raise FileExistsError(manifest)
            if not manifest.exists():
                manifest.write_text(json.dumps(self.protocol,indent=2,sort_keys=True)+"\n")
            print("[POINT100] "+json.dumps(self.protocol,sort_keys=True),flush=True)

    def train_or_validate(self,epoch,training):
        self.net.train(training)
        if self.shared_depth:
            self.module.depth_net.eval()
        loader=self.train_loader if training else self.val_loader
        sampler=self.train_sampler if training else self.val_sampler
        if sampler is not None:
            sampler.set_epoch(epoch)
        limit=(FULL.max_train_batches if training else FULL.max_val_batches)
        totals=torch.zeros(4,device=self.device,dtype=torch.float64)
        for step,batch in enumerate(loader):
            if limit and step>=limit:
                break
            batch=move_to_gpu(batch,self.device,FULL.point_head)
            if training:
                output=self.net(batch)
                loss,output=self.loss_fn(output)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite loss {FULL.point_head} e{epoch} b{step}")
                if output["depth_map_used_for_geometry"]!="depth_net_pred":
                    raise RuntimeError("Geometry is not detached DPT depth")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.0)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    output=self.net(batch)
                    loss,output=self.loss_fn(output)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"Nonfinite validation e{epoch} b{step}")

            vals=torch.stack((
                loss.detach().double(),
                output["A: DepthReg Loss"].detach().double(),
                output["A: Grasp Loss"].detach().double(),
                output["C: Valid Points"].detach().double(),
            ))
            totals[:4]+=vals
            if self.main and training and (step+1)%25==0:
                print(f"[POINT100] {FULL.point_head} e{epoch} step {step+1}/{len(loader)} "
                      f"loss={loss.item():.5f} grasp_valid={float(vals[-1]):.1f}",
                      flush=True)
        nstep=(min(len(loader),limit) if limit else len(loader))
        count=torch.tensor(float(nstep),dtype=torch.float64,device=self.device)
        if self.distributed:
            torch.distributed.all_reduce(totals)
            torch.distributed.all_reduce(count)
        if count.item()<1:
            raise RuntimeError("No train/validation steps")
        result=(totals/count).tolist()
        if self.main:
            print(f"[{'train' if training else 'seen_val'}] {FULL.point_head} "
                  f"e{epoch}: total={result[0]:.6f}, depth={result[1]:.6f}, "
                  f"grasp={result[2]:.6f}, matched_centers={result[3]:.2f}",flush=True)
        return result

    def checkpoint(self,epoch):
        if not self.main:
            return
        data={
            "epoch":epoch+1,
            "completed_epoch":epoch,
            "architecture":"economicgrasp_depth_baseline",
            "point_head":FULL.point_head,
            "point_use_cdf":FULL.point_head=="cdf",
            "pose_depth_mode":"global_film",
            "model_state_dict":self.module.state_dict(),
            "optimizer_state_dict":self.optimizer.state_dict(),
            "full100_protocol":self.protocol,
        }
        if epoch % cfgs.ckpt_save_interval==0 or epoch in (19,cfgs.max_epoch-1):
            path=self.directory/f"checkpoint_epoch_{epoch:03d}.tar"
            torch.save(data,path)
            print(f"[CKPT] {path}",flush=True)
        torch.save(data,self.directory/"checkpoint_latest.tar")

    def run(self):
        history=self.directory/"point_depth_full100_history.jsonl"
        for epoch in range(self.start_epoch,cfgs.max_epoch):
            rate=cfgs.learning_rate*(1.0+math.cos(math.pi*epoch/cfgs.max_epoch))/2.0
            for group in self.optimizer.param_groups:
                group["lr"]=rate
            train=self.train_or_validate(epoch,True)
            seen=self.train_or_validate(epoch,False)
            if self.main:
                with history.open("a",encoding="utf-8") as f:
                    f.write(json.dumps({"epoch":epoch,"train":train,
                                        "seen_validation":seen,"lr":rate})+"\n")
            self.checkpoint(epoch)


def main():
    trainer=None
    try:
        trainer=Trainer()
        trainer.run()
    finally:
        if trainer is not None:
            dist_utils.cleanup_distributed(trainer.distributed)


if __name__=="__main__":
    main()
