#!/usr/bin/env python3
"""Fresh mixed-domain point-centric EconomicGrasp depth baseline.

Same RGB->DAV2/DPT global_film metric depth and scene/frame data allocation as
the current RGB-only CVA study, but the ORIGINAL sparse point-cloud backbone,
view/angle/depth/score/width heads and their legacy losses are intentionally kept.
This is NOT a CVA or CDF model and does not load their grasp checkpoints.

e19 is the predeclared official test checkpoint; validation is Original Seen only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _extract_mix_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gntrans_rgb_root", required=True)
    parser.add_argument("--mix_train_fraction", type=float, default=0.1)
    parser.add_argument("--mix_eval_fraction", type=float, default=0.1)
    parser.add_argument("--gntrans_fused_background", action="store_true")
    args, other = parser.parse_known_args()
    sys.argv = [sys.argv[0], *other]
    return args


MIX = _extract_mix_args()

import numpy as np
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Subset

import train_ddp as ddp
from dataset.graspnet_dataset import GraspNetMultiDataset, GraspNetTransDataset, collate_fn
from models.economicgrasp_depth import economicgrasp_depth_baseline
from models.loss_economicgrasp_depth import get_loss as get_point_loss
from utils.arguments import cfgs
from utils.point_depth_mixed_contract import (
    assert_paired_frames,
    scene_frame_signature,
    scene_stride_indices,
)

CONTRACT_VERSION = 1
MAIN_PARENT = "52d09f925059bec3643610ecf1f1722894627ee5"


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _make_dataset(split, *, original=True):
    common = dict(
        camera=cfgs.camera,
        split=split,
        num_points=cfgs.num_point,
        voxel_size=cfgs.voxel_size,
        remove_outlier=True,
        augment=False,
        load_label=True,
        min_depth=cfgs.min_depth,
        max_depth=cfgs.max_depth,
        bin_num=cfgs.bin_num,
        depth_strides=1,
    )
    if original:
        return GraspNetMultiDataset(
            cfgs.dataset_root,
            use_gt_depth=False,
            use_fuse_depth=bool(cfgs.use_fuse_depth),
            graspness_mode="scene",
            extend_angle=False,
            load_grasp_payload=True,
            **common,
        )
    trans = GraspNetTransDataset(
        cfgs.dataset_root,
        MIX.gntrans_rgb_root,
        use_gt_depth=True,  # historical GN-Trans virtual point/label path
        **common,
    )
    # Default: full rendered GN-Trans depth (matches RayRoPE Mixed-v2).
    # Explicit opt-in reproduces the earlier mixed-trainer fused background.
    if MIX.gntrans_fused_background:
        trans.use_fuse_depth = True
        trans.fusedepthpath = [
            os.path.join(
                cfgs.dataset_root, "tsdf_depth", scene, cfgs.camera,
                f"{int(frame):04d}_depth.png"
            )
            for scene, frame in zip(trans.scenename, trans.frameid)
        ]
    return trans


def _dataset_and_manifest():
    clean_train = _make_dataset("train", original=True)
    trans_train = _make_dataset("train", original=False)
    clean_seen = _make_dataset("test_seen", original=True)
    it = scene_stride_indices(clean_train, MIX.mix_train_fraction)
    jt = scene_stride_indices(trans_train, MIX.mix_train_fraction)
    iv = scene_stride_indices(clean_seen, MIX.mix_eval_fraction)
    assert_paired_frames(clean_train, trans_train, it, jt, "mixed training")
    if MIX.mix_train_fraction == 0.1 and (len(it), len(jt)) != (2600, 2600):
        raise RuntimeError(f"Expected 2600 samples/domain, got {len(it)}, {len(jt)}")
    if MIX.mix_eval_fraction == 0.1 and len(iv) != 780:
        raise RuntimeError(f"Expected 780 Original Seen validation frames, got {len(iv)}")
    train = ConcatDataset([Subset(clean_train, it), Subset(trans_train, jt)])
    val = Subset(clean_seen, iv)
    manifest = dict(
        version=CONTRACT_VERSION,
        architecture="economicgrasp_depth_baseline",
        point_backbone="original TDUnet + EconomicGrasp view/angle/depth/score/width",
        model_input="RGB plus camera pose/K with image pixel indices; predicted depth only for geometry",
        pose_depth_mode=str(cfgs.pose_depth_mode),
        depth_source_for_grasp="predicted_DPT_detached",
        grayscale_or_observed_depth_network_input=False,
        grasp_checkpoint_initialization="fresh",
        pretrained_backbone="local_depth_anything_v2_vitb",
        supervision="standard economic_grasp_label_300views + legacy point grasp loss",
        original_depth_target="rendered_object_plus_TSDF_background" if cfgs.use_fuse_depth else "rendered_full",
        gntrans_depth_target="rendered_object_plus_TSDF_background" if MIX.gntrans_fused_background else "rendered_full",
        original_point_sampling="sensor_depth_workspace",
        gntrans_point_sampling="virtual_rendered_depth_workspace",
        labels_from="standard economic_grasp_label_300views",
        checkpoint_selection="e19 fixed, no Novel or GN-Trans validation selection",
        original_train_count=len(it),
        gntrans_train_count=len(jt),
        train_count=len(train),
        original_seen_validation_count=len(iv),
        original_train_sha256=scene_frame_signature(clean_train,it),
        gntrans_train_sha256=scene_frame_signature(trans_train,jt),
        original_seen_sha256=scene_frame_signature(clean_seen,iv),
        train_fraction=MIX.mix_train_fraction,
        eval_fraction=MIX.mix_eval_fraction,
        train_dataset_root=str(Path(cfgs.dataset_root).resolve()),
        gntrans_rgb_root=str(Path(MIX.gntrans_rgb_root).resolve()),
        seed=int(cfgs.seed),
        max_epochs=int(cfgs.max_epoch),
        local_batch_size=int(cfgs.batch_size),
        max_depth=float(cfgs.max_depth),
        min_depth=float(cfgs.min_depth),
        depth_loss_weight=float(cfgs.depth_prob_loss_weight),
        optimizer="AdamW",
        main_parent=MAIN_PARENT,
        source_sha=_git_sha(),
    )
    return train, val, manifest


class PointDepthMixedTrainer:
    def __init__(self):
        if cfgs.camera != "realsense":
            raise ValueError("The controlled comparison requires --camera realsense.")
        if str(cfgs.pose_depth_mode) != "global_film":
            raise ValueError("This experiment requires --pose_depth_mode global_film.")
        if cfgs.use_gt_depth or cfgs.use_obs_depth or cfgs.use_depth_comp:
            raise ValueError("No observed/GT depth may be used as network geometry.")
        if cfgs.extend_angle or cfgs.use_cdf:
            raise ValueError("Point baseline uses standard economic grasp labels, not CVA/CDF.")
        if cfgs.pin_memory:
            raise ValueError("Do not --pin_memory for variable-size object label lists.")
        if cfgs.checkpoint_path and not cfgs.resume:
            raise ValueError("Fresh training required; checkpoint_path allowed only with --resume.")
        if cfgs.resume and not cfgs.checkpoint_path:
            raise ValueError("--resume requires --checkpoint_path.")
        if cfgs.max_depth != 1.0 or cfgs.min_depth != 0.2:
            raise ValueError("Expected identical metric depth range [0.2, 1.0] m.")
        self.distributed, self.rank, self.local_rank, self.world_size, self.device = (
            ddp.setup_distributed()
        )
        self.main = self.rank == 0
        ddp.seed_everything(cfgs.seed, self.rank)
        self.output = Path(cfgs.log_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.train_dataset, self.val_dataset, self.manifest = _dataset_and_manifest()
        self.manifest["world_size"] = self.world_size
        self.manifest["global_batch_size"] = cfgs.batch_size * self.world_size
        # torch DistributedSampler pads 5200 to 5202 across 3 ranks; log honestly.
        self.manifest["train_examples_per_rank_with_sampler_padding"] = (
            (len(self.train_dataset) + self.world_size - 1) // self.world_size
        )

        self.train_sampler = (
            DistributedSampler(self.train_dataset, shuffle=True,
                               num_replicas=self.world_size, rank=self.rank, seed=cfgs.seed)
            if self.distributed else None
        )
        self.val_sampler = (
            DistributedSampler(self.val_dataset, shuffle=False,
                               num_replicas=self.world_size, rank=self.rank)
            if self.distributed else None
        )
        train_workers = max(0, int(cfgs.num_workers))
        val_workers = max(0, int(cfgs.eval_num_workers))
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=cfgs.batch_size,
            shuffle=self.train_sampler is None, sampler=self.train_sampler,
            num_workers=train_workers, worker_init_fn=ddp.my_worker_init_fn,
            collate_fn=collate_fn, pin_memory=False, drop_last=False,
            persistent_workers=train_workers>0,
            **({"prefetch_factor":1} if train_workers>0 else {}),
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=cfgs.batch_size,
            shuffle=False, sampler=self.val_sampler,
            num_workers=val_workers, worker_init_fn=ddp.my_worker_init_fn,
            collate_fn=collate_fn, pin_memory=False, drop_last=False,
            persistent_workers=False,
            **({"prefetch_factor":1} if val_workers>0 else {}),
        )
        model = economicgrasp_depth_baseline(
            num_points=cfgs.num_point,
            voxel_size=cfgs.voxel_size,
            min_depth=cfgs.min_depth,
            max_depth=cfgs.max_depth,
            is_training=True,
            use_obs_depth=False,
            pose_depth_mode="global_film",
            vis_dir=None,
        ).to(self.device)
        self.net = model
        if self.distributed:
            self.net = DDP(
                model, device_ids=[self.local_rank], output_device=self.local_rank,
                broadcast_buffers=False, find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
        depth_params = [p for p in model.depth_net.parameters() if p.requires_grad]
        depth_ids = {id(p) for p in depth_params}
        grasp_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in depth_ids
        ]
        if not depth_params or not grasp_params:
            raise RuntimeError("Both DPT decoder/FiLM and original 3D grasp network must train")
        self.optimizer = optim.AdamW(
            [
                {"params": depth_params, "weight_decay":cfgs.depth_weight_decay},
                {"params": grasp_params, "weight_decay":cfgs.weight_decay},
            ],
            lr=cfgs.learning_rate,
        )
        self.start_epoch = 0
        if cfgs.resume:
            state = torch.load(cfgs.checkpoint_path, map_location="cpu", weights_only=False)
            if state.get("architecture") != self.manifest["architecture"]:
                raise RuntimeError("Cannot resume unrelated model")
            if state.get("mixed_protocol", {}).get("original_train_sha256") != self.manifest["original_train_sha256"]:
                raise RuntimeError("Mixed train indices do not match resume checkpoint")
            model.load_state_dict(state["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.start_epoch = int(state["epoch"])
        if self.main:
            path = self.output / "point_depth_mixed_protocol.json"
            if path.exists() and not cfgs.resume:
                raise FileExistsError(f"Refuse to overwrite existing protocol: {path}")
            if not path.exists():
                path.write_text(json.dumps(self.manifest, indent=2, sort_keys=True)+"\n")
            print("[MIXED-POINT] "+json.dumps(self.manifest, sort_keys=True), flush=True)

    def unwrap_model(self):
        return self.net.module if isinstance(self.net, DDP) else self.net

    def one_epoch(self, epoch, training):
        self.net.train(training)
        loader = self.train_loader if training else self.val_loader
        sampler = self.train_sampler if training else self.val_sampler
        if sampler is not None:
            sampler.set_epoch(epoch)
        running = torch.zeros(3, dtype=torch.float64, device=self.device)
        count = torch.zeros((), dtype=torch.float64, device=self.device)
        start = time.monotonic()
        for step, batch in enumerate(loader):
            batch = ddp.move_batch_to_device(batch, self.device, non_blocking=False)
            if "gt_depth_m" not in batch or "camera_pose_vec" not in batch:
                raise RuntimeError("Missing supervised metric depth or pose metadata")
            if training:
                outputs = self.net(batch)
                loss, outputs = get_point_loss(outputs)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite point baseline loss epoch={epoch}, batch={step}")
                if outputs.get("depth_map_used_for_geometry") != "depth_net_pred":
                    raise RuntimeError("Backprojection must use detached DPT prediction")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    outputs = self.net(batch)
                    loss, outputs = get_point_loss(outputs)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite validation loss epoch={epoch}, batch={step}")
            running += torch.stack((
                loss.detach().double(),
                outputs["A: DepthReg Loss"].detach().double(),
                outputs["A: Grasp Loss"].detach().double(),
            ))
            count += 1
            if self.main and training and (step+1)%20 == 0:
                print(f"[train] epoch={epoch} step={step+1}/{len(loader)} "
                      f"loss={loss.item():.5f} depth={outputs['A: DepthReg Loss'].item():.5f} "
                      f"elapsed_s={time.monotonic()-start:.1f}", flush=True)
        if self.distributed:
            torch.distributed.all_reduce(running)
            torch.distributed.all_reduce(count)
        if count.item() <= 0:
            raise RuntimeError("No batches in mixed baseline loader")
        values=(running/count).tolist()
        if self.main:
            print(f"[{'TRAIN' if training else 'ORIGINAL-SEEN-VAL'}] "
                  f"epoch={epoch} overall={values[0]:.6f} "
                  f"depth={values[1]:.6f} grasp={values[2]:.6f}",flush=True)
        return values

    def save_checkpoint(self, epoch):
        if not self.main:
            return
        snapshot = {
            "epoch":epoch+1,
            "completed_epoch":epoch,
            "architecture":"economicgrasp_depth_baseline",
            "pose_depth_mode":"global_film",
            "model_state_dict":self.unwrap_model().state_dict(),
            "optimizer_state_dict":self.optimizer.state_dict(),
            "mixed_protocol":self.manifest,
        }
        # Include e19 even when save interval is 5: this is the fixed AP model.
        if (epoch % cfgs.ckpt_save_interval == 0 or epoch in (19, cfgs.max_epoch-1)):
            path=self.output/f"checkpoint_epoch_{epoch:03d}.tar"
            torch.save(snapshot,path)
            print(f"[CHECKPOINT] {path}",flush=True)
        torch.save(snapshot,self.output/"checkpoint_latest.tar")

    def run(self):
        history=self.output/"point_depth_mixed_history.jsonl"
        for epoch in range(self.start_epoch,cfgs.max_epoch):
            lr=cfgs.learning_rate*(np.cos(np.pi*epoch/cfgs.max_epoch)+1)/2
            for group in self.optimizer.param_groups:
                group["lr"]=float(lr)
            train_loss=self.one_epoch(epoch,training=True)
            val_loss=self.one_epoch(epoch,training=False)
            if self.main:
                with history.open("a") as handle:
                    handle.write(json.dumps({"completed_epoch":epoch,"lr":float(lr),
                                             "train":train_loss,"original_seen_val":val_loss})+"\n")
            self.save_checkpoint(epoch)

    def close(self):
        ddp.cleanup_distributed(self.distributed)


def main():
    trainer=PointDepthMixedTrainer()
    try:
        trainer.run()
    finally:
        trainer.close()


if __name__=="__main__":
    main()
