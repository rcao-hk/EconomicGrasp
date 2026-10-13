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



def historical_identity(root: Path, checksum: str) -> dict:
    """Bind the frozen checkpoint to all three prior accepted official runs.

    Reader state_dict strict-load alone cannot distinguish a replaced e19 model
    from the exact e19 that generated the historical published comparison.
    Source-code/forward parity is a SEPARATE requirement.
    """
    root = Path(root).resolve()
    train = root / "train" / "volume_rel"
    def read(path):
        if not path.is_file():
            raise FileNotFoundError(f"Historical GVAR file is missing: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Invalid historical GVAR protocol: {path}")
        return value
    train_proto = read(train / "gvar_protocol.json")
    mixed = read(train / "gntrans_mix_protocol.json")
    if train_proto.get("gvar_config", {}).get("variant") != "volume_rel":
        raise ValueError("Historical training was not volume_rel")
    if train_proto.get("detach_policy") != {"E": True, "Q": True, "C": True}:
        raise ValueError("Historical full E/Q/C detach contract mismatch")
    for k, want in {"train_original": 2600, "train_gntrans": 2600,
                    "validation_original_seen": 780, "validation_gntrans": 0,
                    "world_size": 3, "per_rank_batch": 3, "global_batch": 9,
                    "seed": 0, "frame_stride": 10}.items():
        if train_proto.get(k) != want:
            raise ValueError(f"Historical training mismatch {k}")
    for k, want in {"original_train_count": 2600, "gntrans_train_count": 2600,
                    "mixed_train_count": 5200, "paired_scene_frame_sampling": True}.items():
        if mixed.get(k) != want:
            raise ValueError(f"Historical mixed sampling mismatch {k}")
    manifest_sources = []
    for split in ("test_seen", "test_similar", "test_novel"):
        folder = root / "eval" / "volume_rel_e19" / split
        path = folder / "gvar_inference_protocol.json"
        proto = read(path)
        summary = read(folder / "gvar_inference_summary.json")
        expected = {"checkpoint_sha256": checksum, "completed_epoch": 19,
                    "gvar_config": train_proto["gvar_config"],
                    "split": split, "camera": "realsense", "batch_size": 3,
                    "selected_samples": 780, "frame_stride": 10,
                    "collision_source": "original_sensor", "collision_thresh": .01,
                    "collision_voxel_size": .01,
                    "network_geometry": "predicted metric depth"}
        for k, want in expected.items():
            if proto.get(k) != want:
                raise ValueError(f"Historical {split} mismatch {k}")
        if (summary.get("complete") is not True or
            summary.get("valid_dump_count") != 780 or
            summary.get("expected_count") != 780):
            raise ValueError(f"Historical {split} inference is incomplete")
        ap = folder / f"ap_{split}_realsense.npy"
        if not ap.is_file():
            raise FileNotFoundError(f"Historical {split} official AP missing: {ap}")
        manifest_sources.append({"split": split, "manifest_sha256": file_sha256(path),
                                 "frame_fingerprint": proto.get("frame_fingerprint")})
    return {"status": "matches_all_historical_volume_rel_e19_evaluations",
            "historical_manifest_sources": manifest_sources,
            "limit": "Does not certify current source-vs-old-server forward parity"}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--report", type=Path)
    p.add_argument("--historical-root", type=Path, help="Require matching old official e19 AP manifests")
    args = p.parse_args()
    torch.set_num_threads(1)
    result = verify(args.checkpoint)
    if args.historical_root is not None:
        result['historical_identity'] = historical_identity(args.historical_root, result['checkpoint_sha256'])
    if args.report is not None:
        report_text = json.dumps(result, indent=2) + "\n"
        if args.report.exists():
            if args.report.read_text(encoding="utf-8") != report_text:
                p.error(f"Existing preflight report differs; use a fresh OUT_ROOT: {args.report}")
        else:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open('x', encoding='utf-8') as handle:
                handle.write(report_text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
