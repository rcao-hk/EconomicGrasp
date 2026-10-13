#!/usr/bin/env python3
"""P0-A checkpoint-level CPU preflight for a frozen volume_rel reader.

This validates the actual e19 reader state_dict and R0--R4 dispatch without
loading GraspNet, a CUDA extension, or a training dataset. It does NOT replace
full-network parity against the historically validated server implementation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from models.gripper_volume_reader import GVARConfig, GripperVolumeReader
from utils.gvar_runtime import validate_checkpoint

PREFIX = "kview_grasp_module.decoder.reader."


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _fixture(feat_dim: int, hidden_dim: int):
    """Deterministic camera-projection stress: central and near-boundary probes."""
    torch.manual_seed(20261013)
    H, W = 64, 64
    q = torch.randn(1, 4, 1, 4, hidden_dim)
    centers = torch.tensor([[[0., 0., .50], [.09, 0., .49],
                             [-.09, .02, .49], [0., .09, .50]]])
    rotations = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 4, 1, 1)
    K = torch.tensor([[[180., 0., 31.5],
                       [0., 180., 31.5],
                       [0., 0., 1.]]])
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H),
                            torch.linspace(-1, 1, W), indexing="ij")
    depth = (.52 + .02 * xx + .01 * yy)[None, None]
    context = {"pregeom": torch.randn(1, feat_dim, 16, 16),
               "centers": centers, "rotations": rotations, "K": K,
               "depth": depth, "image_hw": (H, W)}
    return q, context


def verify(checkpoint: Path) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = validate_checkpoint(ck)
    if cfg.variant != "volume_rel" or ck.get("completed_epoch") != 19:
        raise ValueError("P0-A requires frozen volume_rel completed_epoch=19")
    full = ck.get("model_state_dict")
    if not isinstance(full, dict):
        raise ValueError("Expected a full model_state_dict")
    subset = {key[len(PREFIX):]: value for key, value in full.items()
              if key.startswith(PREFIX)}
    wfeat = subset.get("feature_projection.weight")
    wquery = subset.get("query_projection.weight")
    if not torch.is_tensor(wfeat) or not torch.is_tensor(wquery):
        raise ValueError("Reader weights missing; not a full e19 GVAR checkpoint")
    if wfeat.ndim != 4 or wquery.ndim != 2:
        raise ValueError("Reader weight shapes invalid")
    reader = GripperVolumeReader(int(wfeat.shape[1]), int(wquery.shape[1]), cfg)
    reader.load_state_dict(subset, strict=True)
    reader.eval()
    q, context = _fixture(int(wfeat.shape[1]), int(wquery.shape[1]))
    before = {k: value.clone() for k, value in reader.state_dict().items()}
    with torch.no_grad():
        r0_default = reader(q, context)
        reader.set_counterfactual("r0")
        r0_explicit = reader(q, context)
        torch.testing.assert_close(r0_default, r0_explicit, rtol=0, atol=0)
        deltas = {}
        for mode in ("r1", "r2", "r3", "r4"):
            reader.set_counterfactual(mode)
            result = reader(q, context)
            if not bool(torch.isfinite(result).all()):
                raise FloatingPointError("Nonfinite checkpoint reader output: " + mode)
            deltas[mode] = float((result - r0_default).abs().max())
        reader.set_counterfactual("r4")
        r4 = reader(q, context)
        hybrid = GripperVolumeReader(int(wfeat.shape[1]), int(wquery.shape[1]),
                                      replace(cfg, variant="volume_hybrid"))
        hybrid.load_state_dict(subset, strict=True)
        hybrid.eval()
        transplanted = hybrid(q, context)
        torch.testing.assert_close(r4, transplanted, rtol=0, atol=0)
        reader.set_counterfactual("r0")
        torch.testing.assert_close(reader(q, context), r0_default, rtol=0, atol=0)
    for key, old in before.items():
        if not torch.equal(old, reader.state_dict()[key]):
            raise RuntimeError("Reader parameter/buffer changed during preflight: " + key)
    return {
        "status": "reader_checkpoint_preflight_passed",
        "scope": "reader-only; full model/server parity and official AP still required",
        "checkpoint_sha256": file_sha256(checkpoint),
        "reader_variant": cfg.variant,
        "completed_epoch": ck["completed_epoch"],
        "reader_parameter_count": sum(p.numel() for p in reader.parameters()),
        "counterfactual_max_abs_delta_from_r0": deltas,
        "r0_replay_max_abs": 0.0,
        "r4_hybrid_same_weights_max_abs": 0.0,
        "all_reader_state_unchanged": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    result = verify(args.checkpoint)
    if args.report is not None:
        if args.report.exists():
            p.error(f"Refuse to overwrite existing preflight report: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
