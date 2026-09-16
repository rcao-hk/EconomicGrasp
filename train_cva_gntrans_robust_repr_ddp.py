#!/usr/bin/env python3
"""Train GN-Trans mixed CVA-CDF with robust local evidence support.

This keeps the existing 10% GraspNet + 10% GN-Trans training protocol and all
supervision/heads unchanged. The only intervention is the local evidence support
used by the CVA grouping:

  metric : exact existing grouping (controlled rerun)
  wide   : same metric support, wider receptive-field control
  dual   : fixed token budget split between metric-local and image-local support
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch


def _parse_robust_flags():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--robust_support_mode",
        choices=("metric", "wide", "dual"),
        default="dual",
    )
    p.add_argument("--robust_wide_scale", type=float, default=1.5)
    p.add_argument("--robust_image_radius_px", type=float, default=32.0)
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


ROBUST = _parse_robust_flags()

# Install the grouping before the EconomicGrasp model is constructed. The CVA
# wrappers resolve ViewConditionedAttentionGrouping from this module namespace at
# runtime, so no baseline source file needs to be forked.
from models.robust_grasp_support import install_robust_support_grouping

SPEC = install_robust_support_grouping(
    mode=ROBUST.robust_support_mode,
    wide_scale=ROBUST.robust_wide_scale,
    image_radius_px=ROBUST.robust_image_radius_px,
)

import train_cva_gntrans_mix_ddp as mix
from utils.arguments import cfgs


class RobustRepresentationTrainer(mix.GNTransMixedTrainer):
    def __init__(self):
        super().__init__()
        self.robust_repr_protocol = {
            "experiment": "gntrans-mixed-cva-cdf-robust-representation",
            "support_mode": SPEC.mode,
            "wide_scale": float(SPEC.wide_scale),
            "image_radius_px": float(SPEC.image_radius_px),
            "token_budget_changed": False,
            "new_trainable_parameters": False,
            "proposal_center_changed": False,
            "supervision_changed": False,
            "decoder_changed": False,
            "mixed_protocol": self.mix_protocol,
        }
        if self.main:
            self.log_string(
                "[ROBUST-REPR] "
                + json.dumps(self.robust_repr_protocol, sort_keys=True)
            )
            Path(cfgs.log_dir, "robust_repr_protocol.json").write_text(
                json.dumps(
                    self.robust_repr_protocol,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    def save_checkpoint(self, epoch, save_interval=False):
        if not self.main:
            return
        save_dict = {
            "epoch": epoch + 1,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_state_dict": self.unwrap_model().state_dict(),
            "gntrans_mix_contract_version": mix.MIX_CONTRACT_VERSION,
            "gntrans_mix_protocol": self.mix_protocol,
            "robust_repr_protocol": self.robust_repr_protocol,
            "robust_support_mode": SPEC.mode,
            "robust_wide_scale": float(SPEC.wide_scale),
            "robust_image_radius_px": float(SPEC.image_radius_px),
            "use_cdf": bool(self.use_cdf),
            "pose_depth_mode": str(
                getattr(cfgs, "pose_depth_mode", "none")
            ),
            "use_fuse_depth": bool(
                getattr(cfgs, "use_fuse_depth", False)
            ),
        }
        if save_interval:
            torch.save(
                save_dict,
                os.path.join(
                    cfgs.log_dir,
                    f"checkpoint_{epoch}.tar",
                ),
            )
        torch.save(
            save_dict,
            os.path.join(cfgs.log_dir, "checkpoint.tar"),
        )


def main():
    trainer = RobustRepresentationTrainer()
    try:
        trainer.train(trainer.start_epoch)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
