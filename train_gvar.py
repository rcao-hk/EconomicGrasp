#!/usr/bin/env python3
"""GVAR: supervised mixed training; original Seen-only validation; no repair/KD."""
from __future__ import annotations

from functools import partial
import json
import math
import os
from pathlib import Path
import random
import time

from utils.gvar_runtime import (parse_gvar_args, config_from_args, atomic_json, finite_metrics,
    architecture_config, validate_checkpoint, LimitedLoader, BASE_MAIN_SHA, CONTRACT_VERSION)

G = parse_gvar_args()  # before the legacy argparse singleton
CONFIG = config_from_args(G)
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import train_cva_gntrans_mix_ddp as mix
from utils.arguments import cfgs
from models.economicgrasp_gvar import EconomicGraspGVAR


def check_contract():
    if abs(mix.MIX.mix_train_fraction - 0.1) > 1e-10 or abs(mix.MIX.mix_eval_fraction - 0.1) > 1e-10:
        raise ValueError("Use 0.1 for both per-domain training and evaluation fractions")
    if not cfgs.use_cdf or not cfgs.multi_modal or not cfgs.extend_angle:
        raise ValueError("Pass --multi_modal --use_cdf --extend_angle")
    if cfgs.pose_depth_mode != "global_film" or cfgs.kview_mode != "A1":
        raise ValueError("Controlled GVAR protocol is global_film + A1")
    if any(bool(getattr(cfgs, k, False)) for k in ("use_obs_depth", "use_gt_depth", "use_depth_comp", "use_top4_view_infer")):
        raise ValueError("No observed/GT network depth, compensation, or top4 in this experiment")
    old = None
    if cfgs.checkpoint_path:
        if not cfgs.resume:
            raise ValueError("GVAR starts fresh. --checkpoint_path is only allowed with --resume for the SAME variant")
        old = torch.load(cfgs.checkpoint_path, map_location="cpu", weights_only=False)
        validate_checkpoint(old, CONFIG)
        if old["architecture_config"] != architecture_config(cfgs):
            raise ValueError("Resume architecture/config differs from saved checkpoint")
    elif cfgs.resume:
        raise ValueError("--resume requires --checkpoint_path")
    target = Path(cfgs.log_dir)
    if old is None and any((target / name).exists() for name in ("gvar_protocol.json", "gvar_epochs.jsonl", "checkpoint_latest.tar")):
        raise FileExistsError(f"Refusing to append a new experiment to nonempty {target}; choose a new output root")
    return old


class GVARTrainer(mix.GNTransMixedTrainer):
    def __init__(self, resumed):
        self._stats, self._counts = {}, {}
        super().__init__()
        if (len(self.ORIG_TRAIN), len(self.TRANS_TRAIN), len(self.ORIG_VAL)) != (2600, 2600, 780):
            raise ValueError("GVAR requires 2600 original + 2600 GN-Trans train and 780 Original Seen val frames")
        # Do not use GN-Trans, Similar or Novel to select checkpoints.
        self.TEST_DATASET = self.ORIG_VAL
        self.test_sampler = DistributedSampler(self.TEST_DATASET, num_replicas=self.world_size,
            rank=self.rank, shuffle=False, drop_last=False) if self.distributed else None
        self.TEST_DATALOADER = DataLoader(self.TEST_DATASET, batch_size=cfgs.batch_size,
            sampler=self.test_sampler, shuffle=False, num_workers=cfgs.eval_num_workers,
            worker_init_fn=mix.base.my_worker_init_fn, collate_fn=mix.collate_fn, pin_memory=False)
        self.protocol = {
            "base_main_sha": BASE_MAIN_SHA, "gvar_config": CONFIG.to_dict(),
            "architecture_config": architecture_config(cfgs), "detach_policy": {"E": True, "Q": True, "C": True},
            "train_original": 2600, "train_gntrans": 2600, "validation_original_seen": 780,
            "validation_gntrans": 0, "frame_stride": 10, "initialization": "fresh pretrained DINO/DPT task initialization",
            "resume": bool(cfgs.resume), "test_domain": "original GraspNet only", "world_size": self.world_size,
            "per_rank_batch": cfgs.batch_size, "global_batch": cfgs.batch_size * self.world_size,
            "train_steps_per_epoch": len(self.TRAIN_DATALOADER), "smoke_max_batches": G.gvar_max_batches,
            "seed": cfgs.seed, "cli_config": vars(cfgs), "label_contract": "unchanged main CVA nearest-support CDF/width; NOT exact arbitrary-action labels",
            "network_geometry": "predicted metric depth", "depth_assisted_dataset_preprocessing": True,
            "trainable_parameters": sum(p.numel() for p in self.net.parameters() if p.requires_grad),
            "all_parameters": sum(p.numel() for p in self.net.parameters()),
        }
        self.mix_protocol.update({"model_intervention": "GVAR " + CONFIG.variant,
            "mixed_seen_count": 780, "validation_domains": ["original"], "gntrans_seen_count": 0})
        if self.main:
            atomic_json(Path(cfgs.log_dir) / "gvar_protocol.json", self.protocol)
            atomic_json(Path(cfgs.log_dir) / "gntrans_mix_protocol.json", self.mix_protocol)
            self.log_string("[GVAR] " + json.dumps(self.protocol, sort_keys=True))
        if G.gvar_max_batches:
            self.TRAIN_DATALOADER = LimitedLoader(self.TRAIN_DATALOADER, G.gvar_max_batches)
            self.TEST_DATALOADER = LimitedLoader(self.TEST_DATALOADER, G.gvar_max_batches)
        self.best_val = float(resumed.get("best_val_loss", float("inf"))) if resumed else float("inf")
        if resumed and "rng_states" in resumed:
            if len(resumed["rng_states"]) != self.world_size:
                raise ValueError("Resume world_size must match saved run")
            rng = resumed["rng_states"][self.rank]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(rng["cuda"], self.device)

    def extract_scalar_metrics(self, end_points):
        metrics = super().extract_scalar_metrics(end_points)
        n = int(end_points["img"].shape[0])
        for key, value in finite_metrics(metrics).items():
            self._stats[key] = self._stats.get(key, 0.) + n * value
            self._counts[key] = self._counts.get(key, 0) + n
        return metrics

    def phase(self, method, epoch):
        self._stats, self._counts = {}, {}
        started = time.perf_counter()
        value = method(epoch)
        metrics = mix.base.reduce_metric_sums_counts(self._stats, self._counts, self.device, self.distributed)
        return value, finite_metrics(metrics), time.perf_counter() - started

    def train(self, start_epoch):
        for epoch in range(start_epoch, cfgs.max_epoch):
            mix.base.EPOCH_CNT = epoch
            # Replace main's entropy-seeded np.random.seed() with reproducible epoch seeds.
            np.random.seed(int(cfgs.seed) + 1009 * epoch + self.rank)
            self.log_string(f"[GVAR] epoch={epoch} variant={CONFIG.variant}")
            train_loss, train_metrics, train_seconds = self.phase(self.train_one_epoch, epoch)
            val_loss, val_metrics, val_seconds = self.phase(self.evaluate_one_epoch, epoch)
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise FloatingPointError("Nonfinite epoch loss; refusing to save as a valid checkpoint")
            improved = val_loss < self.best_val
            self.best_val = min(self.best_val, val_loss)
            rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                   "cuda": torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None}
            rngs = [rng]
            if self.distributed:
                rngs = [None] * self.world_size
                torch.distributed.all_gather_object(rngs, rng)
            if self.main:
                row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train": train_metrics,
                       "val_original_seen": val_metrics, "train_seconds": train_seconds, "val_seconds": val_seconds,
                       "lr": self.optimizer.param_groups[0]["lr"], "best_val_loss": self.best_val}
                with open(Path(cfgs.log_dir) / "gvar_epochs.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                state = {"epoch": epoch + 1, "completed_epoch": epoch,
                    "model_state_dict": self.unwrap_model().state_dict(), "optimizer_state_dict": self.optimizer.state_dict(),
                    "gvar_contract_version": CONTRACT_VERSION, "gvar_config": CONFIG.to_dict(),
                    "architecture_config": architecture_config(cfgs), "detach_policy": self.protocol["detach_policy"],
                    "gvar_protocol": self.protocol, "pose_depth_mode": cfgs.pose_depth_mode, "use_cdf": True,
                    "best_val_loss": self.best_val, "rng_states": rngs}
                def save(name):
                    target = Path(cfgs.log_dir) / name
                    tmp = target.with_name(target.name + ".tmp")
                    torch.save(state, tmp)
                    os.replace(tmp, target)
                save("checkpoint_latest.tar")
                if improved:
                    save("checkpoint_best_val_loss.tar")
                if (epoch + 1) % cfgs.ckpt_save_interval == 0 or epoch == cfgs.max_epoch - 1:
                    save(f"checkpoint_epoch_{epoch:03d}.tar")
            if self.distributed:
                torch.distributed.barrier()


def main():
    resumed = check_contract()
    mix.base.economicgrasp_dpt = partial(EconomicGraspGVAR, gvar_config=CONFIG)
    trainer = GVARTrainer(resumed)
    try:
        trainer.train(trainer.start_epoch)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
