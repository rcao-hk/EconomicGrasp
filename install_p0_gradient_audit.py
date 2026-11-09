#!/usr/bin/env python3
"""Deprecated no-op compatibility shim for the original P0 audit installer."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_script", default="train_cva_distill_ddp.py")
    parser.add_argument("--dry_run", type=int, choices=(0, 1), default=1)
    parser.parse_args()
    path = Path("train_cva_distill_p0_gradient_audit.py").resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Copy the revised P0 overlay into the repository."
        )
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"[OK] found dedicated gradient-audit entry point: {path}")
    print("[NO-OP] No training source was modified.")
    print(
        "[NEXT] Run the Stage-2 command with "
        "train_cva_distill_p0_gradient_audit.py instead of "
        "train_cva_distill_ddp.py."
    )


if __name__ == "__main__":
    main()
