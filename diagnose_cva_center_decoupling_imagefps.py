#!/usr/bin/env python3
"""Image-FPS launcher for the E00/E01/E10/E11 center diagnostic.

The controlled RGB Stage-1 / distillation checkpoints use the deterministic
image-FPS query contract.  This launcher reuses the main diagnostic implementation
but swaps its model constructor for ``economicgrasp_dpt_student``, which forces
``seed_selection_mode='image_fps'`` and predicted RGB metric geometry.
"""
from __future__ import annotations

import diagnose_cva_center_decoupling as diag
from models.economicgrasp_dpt_distill import economicgrasp_dpt_student


diag.economicgrasp_dpt = economicgrasp_dpt_student


if __name__ == "__main__":
    diag.main()
