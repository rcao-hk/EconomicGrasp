#!/usr/bin/env python3
"""Extract frozen P5 evidence and safe/repairable query outcomes.

The script never updates model weights.  It supports native and deterministic
corrupted geometry conditions and exports compact query-level shards for later
linear-probe and gated-intervention analysis.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.distributed as dist


def _parse_diag_flags():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--diag_split", choices=("train", "test_seen", "test_similar", "test_novel"), required=True)
    p.add_argument("--diag_condition", choices=("native", "corrupt"), required=True)
    p.add_argument("--diag_output_dir", required=True)
    p.add_argument("--diag_query_sample_per_frame", type=int, default=128,
                   help="0 keeps every coherent known query; otherwise deterministic uniform subsample.")
    p.add_argument("--diag_safe_radius_m", type=float, default=0.005)
    p.add_argument("--diag_seed", type=int, default=73191)
    p.add_argument("--diag_corrupt_seed", type=int, default=55117)
    p.add_argument("--diag_flush_frames", type=int, default=32)
    p.add_argument("--diag_max_batches", type=int, default=0)
    args, remaining = p.parse_known_args()
    if args.diag_query_sample_per_frame < 0 or args.diag_flush_frames <= 0 or args.diag_max_batches < 0:
        raise ValueError("Invalid P5 diagnostic sampling/flush/max-batches setting.")
    if args.diag_safe_radius_m < 0:
        raise ValueError("diag_safe_radius_m must be non-negative.")
    sys.argv = [sys.argv[0], *remaining]
    return args


D = _parse_diag_flags()
from utils.p5_runtime import parse_p5_cli, load_p5_model


def _configure_condition(model, condition: str):
    model.train(False)
    if condition == "native":
        model.p5_corrupt_prob = 0.0
        return
    if condition == "corrupt":
        model.p5_corrupt_prob = 1.0
        # Root flag activates structured corruption; child modules remain eval.
        model.training = True
        return
    raise ValueError(condition)


def _accumulate_metric(stats, prefix, metric_dict):
    for name, (total, count) in metric_dict.items():
        key = f"{prefix}/{name}"
        if key not in stats:
            stats[key] = [0.0, 0.0]
        stats[key][0] += float(total.detach().double().item())
        stats[key][1] += float(count.detach().double().item())


def _flush(buffers, root: Path, rank: int, chunk_id: int):
    if not buffers["query_idx"]:
        return chunk_id
    payload = {}
    for key, values in buffers.items():
        payload[key] = np.concatenate(values, axis=0)
        values.clear()
    path = root / f"rank{rank:02d}_chunk{chunk_id:05d}.npz"
    np.savez_compressed(path, **payload)
    return chunk_id + 1


def main():
    args, cfg = parse_p5_cli(training=False)
    if cfg.batch_size != 1:
        raise ValueError("P5 separability extraction currently requires --batch_size 1 for GPU-count invariant corruption.")
    if D.diag_condition == "corrupt" and int(D.diag_corrupt_seed) < 0:
        raise ValueError("diag_corrupt_seed must be non-negative.")

    from utils.p5_runtime import (
        init_distributed, cleanup_distributed, build_dataset, make_loader,
        move_batch, stride_from_fraction,
    )
    from utils.p5_v11_ops import prepare_repair_to_set_targets, repair_to_set_metric_sums
    from utils.p5_separability import build_feature_views, build_query_outcomes

    rank, world, device = init_distributed(cfg.seed)
    out_root = Path(D.diag_output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        model, checkpoint, _ = load_p5_model(args, cfg, device, require_p5=True)
        if int(checkpoint.get("p5_protocol_version", 1)) < 2:
            print("[P5-DIAG] warning: checkpoint predates v1.1 protocol metadata; diagnostic still uses v1.1 target semantics.", flush=True)
        _configure_condition(model, D.diag_condition)

        base, dataset, indices = build_dataset(
            cfg, D.diag_split, cfg.sample_interval, labels=True
        )
        cfg.eval_num_workers = cfg.num_workers
        loader, _ = make_loader(dataset, indices, cfg, rank, world, training=False)
        shard = indices[rank::world]

        hook_cache = {}
        def evidence_hook(_module, inputs, output):
            hook_cache["keypoint_features"] = inputs[0].detach()
            hook_cache["visible"] = inputs[1].detach()
            hook_cache["evidence"] = output.detach()
        def context_hook(_module, inputs, output):
            hook_cache["context"] = output.detach()
        h1 = model.p5_evidence.register_forward_hook(evidence_hook)
        h2 = model.p5_context.register_forward_hook(context_hook)

        keys = [
            "F0", "F1", "F2", "F3", "F4", "safe", "repairable", "need_repair",
            "beneficial", "gain_m", "native_violation_m", "repaired_violation_m",
            "native_safe", "repaired_safe", "pred_delta_abs_m", "query_idx",
            "frame_idx", "scene_id",
        ]
        buffers = {k: [] for k in keys}
        stats = {}
        kept_queries = 0
        processed_frames = 0
        chunk_id = 0

        for step, batch in enumerate(loader):
            if D.diag_max_batches and step >= D.diag_max_batches:
                break
            global_idx = int(shard[step])
            # Make corrupted geometry invariant to rank/world-size for batch=1.
            if D.diag_condition == "corrupt":
                seed = int(D.diag_corrupt_seed) + global_idx
                random.seed(seed); np.random.seed(seed % (2**32))
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            else:
                seed = int(D.diag_seed) + global_idx

            batch = move_batch(batch, device)
            hook_cache.clear()
            with torch.inference_mode():
                ep = model(batch, with_labels=True)
            if not all(k in hook_cache for k in ("keypoint_features", "visible", "evidence", "context")):
                raise RuntimeError("P5 diagnostic hooks did not observe the expected evidence modules.")

            base_targets = {
                "target_delta_local": ep["p5_target_delta_local"],
                "target_known": ep["p5_target_known"],
                "target_distance_m": ep["p5_target_distance_m"],
                "target_utility": ep["p5_target_utility"],
            }
            targets = prepare_repair_to_set_targets(
                base_targets, safe_radius_m=float(D.diag_safe_radius_m)
            )
            feats = build_feature_views(
                keypoint_features=hook_cache["keypoint_features"],
                visible=hook_cache["visible"], evidence=hook_cache["evidence"],
                context=hook_cache["context"], end_points=ep,
                seed_feature_dim=int(model.seed_feature_dim),
                min_depth=float(model.min_depth), max_depth=float(model.max_depth),
            )
            outcomes = build_query_outcomes(ep, targets)

            zero_delta = torch.zeros_like(ep["p5_delta_local"])
            oracle_delta = torch.where(
                targets["target_repairable"].unsqueeze(-1), ep["p5_delta_local"], zero_delta
            )
            _accumulate_metric(stats, "native", repair_to_set_metric_sums(
                zero_delta, ep["p5_rotation"], ep["p5_proposal_center"], targets
            ))
            _accumulate_metric(stats, "p5", repair_to_set_metric_sums(
                ep["p5_delta_local"], ep["p5_rotation"], ep["p5_proposal_center"], targets
            ))
            _accumulate_metric(stats, "oracle_gate", repair_to_set_metric_sums(
                oracle_delta, ep["p5_rotation"], ep["p5_proposal_center"], targets
            ))

            known_idx = torch.where(outcomes["known"][0])[0].detach().cpu()
            if D.diag_query_sample_per_frame and known_idx.numel() > D.diag_query_sample_per_frame:
                gen = torch.Generator(device="cpu").manual_seed(seed + 9973)
                perm = torch.randperm(known_idx.numel(), generator=gen)[:D.diag_query_sample_per_frame]
                known_idx = known_idx[perm]
            idx = known_idx.to(device)
            n = int(idx.numel())
            if n:
                for name in ("F0", "F1", "F2", "F3", "F4"):
                    buffers[name].append(feats[name][0].index_select(0, idx).float().cpu().numpy().astype(np.float16))
                for name in ("safe", "repairable", "need_repair", "beneficial", "native_safe", "repaired_safe"):
                    buffers[name].append(outcomes[name][0].index_select(0, idx).cpu().numpy().astype(np.uint8))
                for name in ("gain_m", "native_violation_m", "repaired_violation_m", "pred_delta_abs_m"):
                    buffers[name].append(outcomes[name][0].index_select(0, idx).float().cpu().numpy().astype(np.float32))
                buffers["query_idx"].append(known_idx.numpy().astype(np.int16))
                buffers["frame_idx"].append(np.full((n,), global_idx, dtype=np.int32))
                scene_name = str(base.scenename[global_idx])
                try:
                    scene_id = int(scene_name.split("_")[-1])
                except Exception:
                    scene_id = -1
                buffers["scene_id"].append(np.full((n,), scene_id, dtype=np.int16))
                kept_queries += n
            processed_frames += 1

            if processed_frames % int(D.diag_flush_frames) == 0:
                chunk_id = _flush(buffers, out_root, rank, chunk_id)
            if rank == 0 and step % 20 == 0:
                print(
                    f"[P5-DIAG] split={D.diag_split} condition={D.diag_condition} "
                    f"step={step}/{len(loader)} kept={kept_queries}", flush=True
                )
            del ep, batch, feats, outcomes, targets, base_targets

        chunk_id = _flush(buffers, out_root, rank, chunk_id)
        h1.remove(); h2.remove()

        # Aggregate sufficient statistics and basic counts across ranks.
        names = sorted(stats)
        packed = torch.tensor([stats[k] for k in names], dtype=torch.float64, device=device)
        counts = torch.tensor([processed_frames, kept_queries], dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(packed)
            dist.all_reduce(counts)
        if rank == 0:
            metric_summary = {}
            for i, name in enumerate(names):
                total, count = packed[i].tolist()
                metric_summary[name] = {
                    "sum": total, "count": count,
                    "mean": (total / count if count > 0 else None),
                }
            summary = {
                "experiment": "p5-safe-repairable-separability-v1",
                "checkpoint": str(Path(cfg.checkpoint_path).resolve()),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "p5_protocol_version": checkpoint.get("p5_protocol_version"),
                "split": D.diag_split,
                "condition": D.diag_condition,
                "sample_fraction": float(cfg.sample_interval),
                "sample_stride": stride_from_fraction(cfg.sample_interval),
                "safe_radius_m": float(D.diag_safe_radius_m),
                "query_sample_per_frame": int(D.diag_query_sample_per_frame),
                "corrupt_seed": int(D.diag_corrupt_seed),
                "processed_frames": int(counts[0].item()),
                "kept_known_queries": int(counts[1].item()),
                "world_size": int(world),
                "unknown_queries_exported": False,
                "metrics": metric_summary,
            }
            (out_root / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
