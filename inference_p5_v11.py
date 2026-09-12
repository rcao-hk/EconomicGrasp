#!/usr/bin/env python3
"""P5-v1.1 inference entrypoint.

P5-v1.1 changes training semantics only; deployment architecture is identical to
P5-v1.  This wrapper therefore reuses the controlled P5 inference path after a
strict checkpoint-protocol check.  No synthetic corruption is active at
inference.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _preflight() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--checkpoint_path", default="")
    p.add_argument("--p5_eval_only", action="store_true")
    args, _ = p.parse_known_args()
    if args.p5_eval_only:
        return
    if not args.checkpoint_path or not Path(args.checkpoint_path).is_file():
        raise FileNotFoundError(
            f"P5-v1.1 checkpoint not found: {args.checkpoint_path}"
        )
    import torch
    ck = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or int(ck.get("p5_protocol_version", -1)) != 2:
        raise ValueError(
            "inference_p5_v11.py requires a P5-v1.1 checkpoint with "
            "p5_protocol_version=2."
        )
    if int(ck.get("p5_smoke_max_batches", 0)) != 0:
        raise ValueError("Official P5-v1.1 inference rejects smoke checkpoints.")


def main() -> None:
    _preflight()
    # Import only after preflight. inference_p5 retains the exact Stage-1/P5
    # decode, translation-only intervention and GraspNet evaluation protocol.
    from inference_p5 import main as p5_main
    p5_main()


if __name__ == "__main__":
    main()
