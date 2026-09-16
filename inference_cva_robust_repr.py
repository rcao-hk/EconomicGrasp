#!/usr/bin/env python3
"""Original-GraspNet inference for robust CVA evidence-support variants."""
from __future__ import annotations

import argparse
import sys

import torch


def _parse_robust_flags():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--robust_support_mode",
        choices=("metric", "wide", "dual"),
        required=True,
    )
    p.add_argument("--robust_wide_scale", type=float, default=1.5)
    p.add_argument("--robust_image_radius_px", type=float, default=32.0)
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


R = _parse_robust_flags()

from models.robust_grasp_support import install_robust_support_grouping

SPEC = install_robust_support_grouping(
    mode=R.robust_support_mode,
    wide_scale=R.robust_wide_scale,
    image_radius_px=R.robust_image_radius_px,
)

import inference_cva as base
from utils.arguments import cfgs


def _check_checkpoint_protocol():
    ck = torch.load(cfgs.checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        return
    saved_mode = ck.get("robust_support_mode")
    if saved_mode is not None and str(saved_mode) != SPEC.mode:
        raise ValueError(
            f"Checkpoint robust_support_mode={saved_mode!r}, "
            f"CLI={SPEC.mode!r}."
        )
    if SPEC.mode == "wide" and ck.get("robust_wide_scale") is not None:
        if abs(float(ck["robust_wide_scale"]) - SPEC.wide_scale) > 1e-9:
            raise ValueError("robust_wide_scale mismatch between checkpoint and CLI.")
    if SPEC.mode == "dual" and ck.get("robust_image_radius_px") is not None:
        if abs(float(ck["robust_image_radius_px"]) - SPEC.image_radius_px) > 1e-9:
            raise ValueError("robust_image_radius_px mismatch between checkpoint and CLI.")


if __name__ == "__main__":
    _check_checkpoint_protocol()
    base.inference()
