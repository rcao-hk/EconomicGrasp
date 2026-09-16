#!/usr/bin/env python3
"""GN-Trans inference for robust CVA evidence-support variants."""
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

# Import only after the grouping symbol is patched and robust-only CLI flags have
# been removed. The base module then parses GN-Trans and repository arguments.
import inference_cva_gntrans as base
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
    saved_scale = ck.get("robust_wide_scale")
    if SPEC.mode == "wide" and saved_scale is not None:
        if abs(float(saved_scale) - float(SPEC.wide_scale)) > 1e-9:
            raise ValueError(
                f"Checkpoint robust_wide_scale={saved_scale}, "
                f"CLI={SPEC.wide_scale}."
            )
    saved_radius = ck.get("robust_image_radius_px")
    if SPEC.mode == "dual" and saved_radius is not None:
        if abs(float(saved_radius) - float(SPEC.image_radius_px)) > 1e-9:
            raise ValueError(
                f"Checkpoint robust_image_radius_px={saved_radius}, "
                f"CLI={SPEC.image_radius_px}."
            )


if __name__ == "__main__":
    _check_checkpoint_protocol()
    base.inference()
