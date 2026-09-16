#!/usr/bin/env python3
"""Mine frozen K-center features and exact-action labels for selector training.

The RGB grasp network is fully frozen.  For each selected image-FPS query and
camera-z hypothesis, this script stores frozen local features, the deployed CDF
score, and the post-hoc CAD/DexNet exact-action label.  The analytic evaluator is
used only during cache mining; selector training consumes the cache offline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", default="train", choices=("train", "test_seen", "test_similar", "test_novel"))
    p.add_argument("--camera", default="realsense")
    p.add_argument("--sample_interval", type=float, default=0.1)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_points", type=int, default=20000)
    p.add_argument("--min_depth", type=float, default=0.2)
    p.add_argument("--max_depth", type=float, default=1.0)
    p.add_argument("--bin_num", type=int, default=256)
    p.add_argument("--pose_depth_mode", default="global_film", choices=("none", "global_film", "ray_gravity_film"))
    p.add_argument("--offsets_mm", default="-40,-20,-10,0,10,20,40")
    p.add_argument("--query_eval_num", type=int, default=64)
    p.add_argument("--query_eval_mode", default="topk_uniform", choices=("all", "topk", "uniform", "topk_uniform"))
    p.add_argument("--fc_mode", default="reuse_contacts", choices=("reuse_contacts", "official"))
    p.add_argument("--verify_n", type=int, default=0)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--noop_check_samples", type=int, default=1)
    p.add_argument("--noop_atol", type=float, default=5e-5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


ARGS = _parse_args()
sys.argv = [sys.argv[0]]

from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
from models.economicgrasp_bip3d import pred_decode_center_view_angle
from models.economicgrasp_dpt_distill import economicgrasp_dpt_student
from utils.arguments import cfgs
from utils.cva_center_decoupling import assert_native_reread_equivalent, rerun_cdf_with_read_center
from utils.ray_bestofk_diagnostic import build_ray_center_hypotheses, friction_utility, parse_offsets_mm
from utils.ray_pairwise_selector import extract_action_conditioned_features


def _configure_cfg():
    cfgs.use_top4_view_infer = False
    cfgs.kview_mode = "A1"
    cfgs.kview_k = 1
    cfgs.use_cdf = True
    cfgs.use_obs_depth = False
    cfgs.pose_depth_mode = ARGS.pose_depth_mode


def _load_checkpoint(model: torch.nn.Module, path: str):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint does not contain a state dict.")
    result = model.load_state_dict(state, strict=False)
    optional = ("rgb_geometry_diagnostics.",)
    missing = [k for k in result.missing_keys if not k.startswith(optional)]
    unexpected = [k for k in result.unexpected_keys if not k.startswith(optional)]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def _subset_indices(total: int, interval: float, max_samples: int, shard_id: int, num_shards: int) -> List[int]:
    if not (0.0 < interval <= 1.0):
        raise ValueError("sample_interval must be in (0,1].")
    if num_shards <= 0 or not (0 <= shard_id < num_shards):
        raise ValueError("Invalid shard_id/num_shards.")
    stride = max(1, int(round(1.0 / interval)))
    out = []
    for start in range(0, total, 256):
        scene_id = start // 256
        if scene_id % num_shards != shard_id:
            continue
        out.extend(range(start, min(start + 256, total), stride))
        if max_samples > 0 and len(out) >= max_samples:
            return out[:max_samples]
    return out


def _move_batch(batch, device):
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=False)
        elif isinstance(v, (list, tuple)):
            raise TypeError(f"Unexpected list-valued batch key {k!r}.")
    return batch


def _uniform_pick(indices: torch.Tensor, count: int) -> torch.Tensor:
    n = int(indices.numel())
    if count <= 0 or count >= n:
        return indices
    pos = torch.round(torch.linspace(0, n - 1, steps=count, device=indices.device)).long()
    pos = torch.unique(pos, sorted=True)
    if pos.numel() < count:
        used = torch.zeros(n, device=indices.device, dtype=torch.bool)
        used[pos] = True
        fill = torch.nonzero(~used, as_tuple=False).squeeze(1)[: count - pos.numel()]
        pos = torch.sort(torch.cat((pos, fill))).values
    return indices.index_select(0, pos[:count])


def _select_queries(native_grasps: torch.Tensor, count: int, mode: str) -> torch.Tensor:
    n = native_grasps.shape[0]
    eligible = torch.arange(n, device=native_grasps.device)
    if count <= 0 or count >= n or mode == "all":
        return eligible
    score = native_grasps[:, 0].float()
    if mode == "topk":
        return torch.argsort(score, descending=True, stable=True)[:count]
    if mode == "uniform":
        return _uniform_pick(eligible, count)
    n_top = count // 2
    ranked = torch.argsort(score, descending=True, stable=True)
    top = ranked[:n_top]
    uniform = _uniform_pick(ranked[n_top:], count - n_top)
    return torch.cat((top, uniform))


def _evaluate_grid(evaluator, scene_id, anno_id, grasps_kn17, valid_kn):
    K, N, _ = grasps_kn17.shape
    flat = grasps_kn17.reshape(K * N, 17)
    valid_flat = valid_kn.reshape(K * N)
    ids = np.flatnonzero(valid_flat)
    result = evaluator.evaluate(scene_id, anno_id, flat[ids])
    friction = np.full(K * N, np.nan, dtype=np.float32)
    collision = np.full(K * N, -1, dtype=np.int8)
    pure_collision = np.full(K * N, -1, dtype=np.int8)
    empty = np.full(K * N, -1, dtype=np.int8)
    friction[ids] = result.friction
    collision[ids] = result.collision_or_empty.astype(np.int8)
    pure_collision[ids] = result.pure_collision.astype(np.int8)
    empty[ids] = result.empty.astype(np.int8)
    return (
        friction.reshape(K, N),
        collision.reshape(K, N),
        pure_collision.reshape(K, N),
        empty.reshape(K, N),
    )


def main():
    _configure_cfg()
    offsets = parse_offsets_mm(ARGS.offsets_mm)
    zero = [i for i, x in enumerate(offsets) if abs(x) < 1e-9]
    if len(zero) != 1:
        raise ValueError("Exactly one 0-mm offset is required.")
    zero_k = zero[0]
    out_root = Path(ARGS.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = GraspNetMultiDataset(
        ARGS.dataset_root,
        split=ARGS.split,
        camera=ARGS.camera,
        num_points=ARGS.num_points,
        remove_outlier=True,
        augment=False,
        load_label=False,
        use_gt_depth=False,
        min_depth=ARGS.min_depth,
        max_depth=ARGS.max_depth,
        bin_num=ARGS.bin_num,
    )
    indices = _subset_indices(len(dataset), ARGS.sample_interval, ARGS.max_samples, ARGS.shard_id, ARGS.num_shards)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=False,
        num_workers=ARGS.num_workers, collate_fn=collate_fn,
        pin_memory=False, persistent_workers=(ARGS.num_workers > 0),
    )

    model = economicgrasp_dpt_student(
        min_depth=ARGS.min_depth, max_depth=ARGS.max_depth, bin_num=ARGS.bin_num,
        is_training=False, use_obs_depth=False, pose_depth_mode=ARGS.pose_depth_mode,
        use_cdf=True, vis_dir=None,
    ).to(device)
    _load_checkpoint(model, ARGS.checkpoint_path)
    model.eval()
    evaluator = ExactGraspNetActionEvaluator(
        ARGS.dataset_root, ARGS.camera, split=ARGS.split,
        fc_mode=ARGS.fc_mode, verify_n=ARGS.verify_n, strict=True,
    )

    mined = skipped = 0
    noop_done = 0
    feature_dim = None
    start = time.time()
    for local_i, batch in enumerate(loader):
        dataset_idx = indices[local_i]
        scene_guess, anno_guess = dataset_idx // 256, dataset_idx % 256
        cache_path = out_root / f"scene_{scene_guess:04d}" / f"ann_{anno_guess:04d}.npz"
        if cache_path.exists() and not ARGS.overwrite:
            skipped += 1
            continue

        batch = _move_batch(batch, device)
        batch["cva_export_angle_feature"] = False
        batch["cva_compute_diagnostics"] = False
        batch["geometry_compute_diagnostics"] = False
        with torch.inference_mode():
            ep = model(batch)
            native_pred = pred_decode_center_view_angle(ep, use_cdf=True)[0]

        scene_id = int(batch["scene_idx"].reshape(-1)[0].item())
        anno_id = int(batch["anno_idx"].reshape(-1)[0].item())
        cache_path = out_root / f"scene_{scene_id:04d}" / f"ann_{anno_id:04d}.npz"
        if cache_path.exists() and not ARGS.overwrite:
            skipped += 1
            continue

        native_xyz = ep["kview_base_xyz_graspable"].float()
        token_idx = ep["kview_base_token_sel_idx"].long()
        H, W = ep["depth_map_used_for_geometry"].shape[-2:]
        centers, center_valid = build_ray_center_hypotheses(
            native_xyz, token_idx, ep["K"], (H, W), offsets, ARGS.min_depth, ARGS.max_depth
        )
        if not bool(center_valid[zero_k].all()):
            raise RuntimeError("Native center became invalid.")

        eval_idx = _select_queries(native_pred, ARGS.query_eval_num, ARGS.query_eval_mode)
        grasps_k = []
        selected_feat_k = []
        mean_feat_k = []
        best_angle_k = []
        with torch.inference_mode():
            for k in range(len(offsets)):
                epk, grouped = rerun_cdf_with_read_center(
                    model, ep, read_center=centers[k], output_center=centers[k]
                )
                if k == zero_k and noop_done < ARGS.noop_check_samples:
                    metrics = assert_native_reread_equivalent(ep, epk, atol=ARGS.noop_atol)
                    replay = pred_decode_center_view_angle(epk, use_cdf=True)[0]
                    decoded_diff = float((replay - native_pred).abs().max().item())
                    if decoded_diff > ARGS.noop_atol:
                        raise RuntimeError(f"No-op decoded mismatch {decoded_diff:.3e}")
                pred = native_pred if k == zero_k else pred_decode_center_view_angle(epk, use_cdf=True)[0]
                sel, mean, angle = extract_action_conditioned_features(grouped, epk)
                grasps_k.append(pred.index_select(0, eval_idx))
                selected_feat_k.append(sel[0].index_select(0, eval_idx))
                mean_feat_k.append(mean[0].index_select(0, eval_idx))
                best_angle_k.append(angle[0].index_select(0, eval_idx))
        if noop_done < ARGS.noop_check_samples:
            noop_done += 1

        grasps = torch.stack(grasps_k, dim=0).detach().cpu().numpy().astype(np.float32)
        selected_feat = torch.stack(selected_feat_k, dim=0).detach().cpu().numpy().astype(np.float16)
        mean_feat = torch.stack(mean_feat_k, dim=0).detach().cpu().numpy().astype(np.float16)
        best_angle = torch.stack(best_angle_k, dim=0).detach().cpu().numpy().astype(np.int16)
        valid_kn = center_valid[:, 0].index_select(1, eval_idx).detach().cpu().numpy().astype(bool)
        query_id = eval_idx.detach().cpu().numpy().astype(np.int64)
        raw_score = grasps[:, :, 0].astype(np.float32)
        friction, collision, pure_collision, empty = _evaluate_grid(
            evaluator, scene_id, anno_id, grasps, valid_kn
        )
        utility = friction_utility(friction)
        native_z = native_xyz[0].index_select(0, eval_idx)[:, 2].detach().cpu().numpy().astype(np.float32)

        feature_dim = int(selected_feat.shape[-1])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            scene_id=np.int32(scene_id), anno_id=np.int32(anno_id), dataset_idx=np.int64(dataset_idx),
            query_id=query_id, offsets_mm=np.asarray(offsets, dtype=np.float32), zero_index=np.int16(zero_k),
            valid=valid_kn, selected_feature=selected_feat, mean_feature=mean_feat,
            best_angle=best_angle, raw_score=raw_score, friction=friction, utility=utility,
            collision_or_empty=collision, pure_collision=pure_collision, empty=empty,
            native_z=native_z,
        )
        os.replace(tmp, cache_path)
        mined += 1
        if mined == 1 or mined % 20 == 0:
            print(
                f"[PAIR-CACHE] shard={ARGS.shard_id}/{ARGS.num_shards} mined={mined} skipped={skipped} "
                f"scene={scene_id:04d} anno={anno_id:04d} K={len(offsets)} N={len(query_id)} "
                f"elapsed={(time.time()-start)/60:.1f}m",
                flush=True,
            )

    protocol = {
        "split": ARGS.split,
        "checkpoint": os.path.abspath(ARGS.checkpoint_path),
        "sample_interval": ARGS.sample_interval,
        "offsets_mm": list(offsets),
        "query_eval_num": ARGS.query_eval_num,
        "query_eval_mode": ARGS.query_eval_mode,
        "shard_id": ARGS.shard_id,
        "num_shards": ARGS.num_shards,
        "mined": mined,
        "skipped": skipped,
        "group_feature_dim": feature_dim,
        "label": "exact CAD/DexNet utility; post-hoc cache only",
    }
    with (out_root / f"protocol_shard{ARGS.shard_id:02d}.json").open("w") as f:
        json.dump(protocol, f, indent=2, sort_keys=True)
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
