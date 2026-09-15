#!/usr/bin/env python3
"""Image-FPS launcher for the E00/E01/E10/E11 center diagnostic.

The controlled RGB Stage-1 / distillation checkpoints use the deterministic
image-FPS query contract. This launcher reuses the main diagnostic implementation
but swaps its model constructor for ``economicgrasp_dpt_student``, which forces
``seed_selection_mode='image_fps'`` and predicted RGB metric geometry.

The canonical ``economicgrasp_dpt_cva_cdf_distill_stage1`` checkpoint uses the
pose-conditioned metric-depth model with ``pose_depth_mode='global_film'``.
Therefore direct invocations of this image-FPS launcher default to global FiLM
unless an explicit ``--pose_depth_mode`` is supplied.
"""
from __future__ import annotations

import sys


if "--pose_depth_mode" not in sys.argv:
    sys.argv.extend(["--pose_depth_mode", "global_film"])

import diagnose_cva_center_decoupling as diag
from models.economicgrasp_dpt_distill import economicgrasp_dpt_student


diag.economicgrasp_dpt = economicgrasp_dpt_student


if __name__ == "__main__":
    diag.main()
