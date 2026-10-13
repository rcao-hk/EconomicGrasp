#!/usr/bin/env python3
"""Frozen U3 diagnostic: alter only halfwidth entering RayRoPE grouping.

Wrapper reuses the original inference_moge_rayrope.py without modifying hashed
model/inference sources. The official evaluator remains eval_moge_rayrope.py.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODES = ("learned", "point", "constant_train", "fixed20", "shuffle")


def training_mean(checkpoint):
    path = Path(checkpoint).parent / "metrics.json"
    rows = json.loads(path.read_text())
    if not rows or rows[-1]["epoch"] != 19:
        raise RuntimeError("Training metrics must end at epoch 19")
    row = rows[-1]["train"]
    val = row.get("scalars", {}).get("interval_mean_halfwidth_m")
    if val is None:
        val = row["interval_width_m"] / 2.
    if not (.001 <= float(val) <= .08):
        raise RuntimeError("Invalid train-only halfwidth")
    return float(val), hashlib.sha256(path.read_bytes()).hexdigest()


def intervention(sigma, depth, mode, frame_ids, mean_width, shuffle_seed):
    if sigma.shape != depth.shape or sigma.ndim != 4 or sigma.shape[1] != 1:
        raise RuntimeError("Expected aligned [B,1,H,W] depth and halfwidth")
    if len(frame_ids) != sigma.shape[0] or not torch.isfinite(sigma).all():
        raise RuntimeError("Invalid sigma/batch")
    if mode == "learned":
        return sigma
    if mode == "point":
        return torch.zeros_like(sigma)
    if mode in ("constant_train", "fixed20"):
        return torch.full_like(sigma, mean_width if mode == "constant_train" else .02)
    if mode != "shuffle":
        raise ValueError(mode)
    out = sigma.clone()
    for b, (scene, anno) in enumerate(frame_ids):
        valid = (torch.isfinite(depth[b, 0]) & (depth[b, 0] > 0)).flatten()
        positions = torch.nonzero(valid).flatten()
        n = positions.numel()
        if n < 2:
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed((shuffle_seed + int(scene) * 1000003
                               + int(anno) * 9176) % (2**63-1))
        perm = torch.randperm(n, generator=generator).to(positions.device)
        values = sigma[b, 0].flatten()[positions]
        out[b, 0].flatten()[positions] = values[perm]
    return out


class Hook:
    """Record upstream proposal hashes and intercept the grouping-only sigma."""

    def __init__(self, mode, mean, seed):
        self.mode, self.mean, self.seed = mode, mean, seed
        self.frames = None
        self.hashes = []
        self.completed = {}
        self.calls = 0
        self.original_sum = 0.
        self.applied_sum = 0.
        self.pixels = 0

    def before_model(self, module, args):
        batch = args[0]
        if self.frames is not None:
            raise RuntimeError("Previous diagnostic forward not completed")
        self.frames = list(zip(batch["scene_idx"].tolist(), batch["anno_idx"].tolist()))
        self.hashes = []
        self.calls = 0

    def before_group(self, module, args, kwargs):
        if self.frames is None or self.calls != 0 or args:
            raise RuntimeError("Unexpected grouping call/contract")
        context = module.geometry_context
        if context is None or "sigma" not in context:
            raise RuntimeError("Missing U3 learned sigma")
        sigma, depth = context["sigma"], kwargs["depth_map"]
        if len(self.frames) != sigma.shape[0]:
            raise RuntimeError("Changed batch size before RayRoPE")
        for b in range(len(self.frames)):
            h = hashlib.sha256()
            for key in ("token_sel_idx", "seed_xyz", "top_view_rot"):
                arr = kwargs[key][b].detach().contiguous().cpu().numpy()
                h.update(key.encode())
                h.update(str(arr.shape).encode())
                h.update(arr.tobytes())
            self.hashes.append(h.hexdigest())
        changed = intervention(sigma, depth, self.mode, self.frames, self.mean, self.seed)
        valid = torch.isfinite(depth) & (depth > 0) & torch.isfinite(sigma)
        self.original_sum += sigma[valid].sum().item()
        self.applied_sum += changed[valid].sum().item()
        self.pixels += valid.sum().item()
        # A fresh context avoids mutating the predicted sigma itself.
        module.geometry_context = dict(context, sigma=changed.detach())
        self.calls += 1

    def after_model(self, module, args, output):
        if self.frames is None or self.calls != 1 or len(self.frames) != len(self.hashes):
            raise RuntimeError("Expected exactly one grouping per model forward")
        self.completed.update(zip(self.frames, self.hashes))
        self.frames = None

    def stats(self):
        return dict(mode=self.mode, calls=len(self.completed),
                    predicted_halfwidth_mean_m=self.original_sum/max(1, self.pixels),
                    applied_halfwidth_mean_m=self.applied_sum/max(1, self.pixels))


def main():
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--mode", choices=MODES, required=True)
    wrapper.add_argument("--native-reference-root", default="")
    wrapper.add_argument("--shuffle-seed", type=int, default=20261013)
    a, rest = wrapper.parse_known_args()

    # The remaining arguments belong to the ORIGINAL inference parser.
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--checkpoint", required=True)
    probe.add_argument("--output-root", required=True)
    c, _ = probe.parse_known_args(rest)
    ckpt = Path(c.checkpoint).resolve()
    native = Path(a.native_reference_root).resolve() if a.native_reference_root else None
    if native and a.mode != "learned":
        raise ValueError("Native parity is valid only for learned mode")
    if a.shuffle_seed < 0:
        raise ValueError("shuffle seed must be nonnegative")

    import moge_rayrope.runtime as runtime
    import inference_moge_rayrope as original
    mean, metrics_sha = training_mean(ckpt)
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    original_load = runtime.load_model
    original_manifest = runtime.ensure_manifest
    original_json = runtime.atomic_json
    original_npy = runtime.atomic_npy
    state = {"hook": None}

    def patched_load(*args, **kwargs):
        model, ck = original_load(*args, **kwargs)
        p = ck["protocol"]
        cfg = p["model"]
        if (cfg.get("use_moge") or not cfg.get("use_rayrope")
            or cfg.get("ray_encoding") != "expected"
            or cfg.get("uncertainty") != "learned"
            or cfg.get("uncertainty_loss") != "laplace_decoupled"
            or p.get("mixed_depth_supervision", {}).get("contract_version") != 2
            or ck["epoch"] != 19 or p.get("partial_run")):
            raise ValueError("Expected final Mixed-v2 U3 checkpoint, no MoGe")
        hook = Hook(a.mode, mean, a.shuffle_seed)
        model.register_forward_pre_hook(hook.before_model)
        model.register_forward_hook(hook.after_model)
        model.base.kview_grasp_module.group.register_forward_pre_hook(
            hook.before_group, with_kwargs=True)
        state["hook"] = hook
        return model, ck

    def patched_manifest(path, value):
        value["diagnostic"] = dict(
            kind="frozen_u3_grouping_halfwidth_v1", mode=a.mode,
            script_sha256=script_sha, train_metrics_sha256=metrics_sha,
            constant_train_halfwidth_m=mean, shuffle_seed=a.shuffle_seed,
            native_reference_root=str(native) if native else None,
            only_grouping_halfwidth_changed=True)
        return original_manifest(path, value)

    def patched_json(path, value):
        path = Path(path)
        if path.parent.name == "completed":
            scene, anno = map(int, path.stem.split("_"))
            h = state["hook"].completed.get((scene, anno))
            if h is None:
                raise RuntimeError(f"Missing upstream proposal hash {scene}:{anno}")
            value["proposal_sha256"] = h
        elif path.name.startswith("shard_") and path.suffix == ".json":
            value["halfwidth_intervention"] = state["hook"].stats()
        return original_json(path, value)

    def patched_npy(path, value):
        if native is not None:
            parts = Path(path).parts
            marker = next((v for v in parts if v.startswith("test_collision_")), None)
            if not marker or "dump" not in parts:
                raise RuntimeError("Unexpected diagnostic dump location")
            tail = Path(*parts[parts.index("dump")+1:])
            reference = native / marker / "dump" / tail
            if not reference.is_file():
                raise FileNotFoundError(reference)
            old = np.load(reference, allow_pickle=False)
            if old.shape != value.shape or not np.allclose(old, value, atol=1e-6, rtol=1e-6):
                raise AssertionError(f"Learned U3 != native N×17 grasp dump: {reference}")
        return original_npy(path, value)

    runtime.load_model = patched_load
    runtime.ensure_manifest = patched_manifest
    runtime.atomic_json = patched_json
    runtime.atomic_npy = patched_npy
    sys.argv = [sys.argv[0], *rest]
    print(f"[U3-DIAG] mode={a.mode}, train-only mean={mean*1000:.4f} mm", flush=True)
    original.main()


if __name__ == "__main__":
    main()
