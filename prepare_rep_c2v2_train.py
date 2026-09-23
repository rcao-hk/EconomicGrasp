#!/usr/bin/env python3
"""Mine Rep-C2-v2 training examples from the real full-path correction problem.

For each Rep-A training frame, this script samples ONE deterministic joint-depth
condition, reruns the real Stage-1 path, lets frozen A1 fixed-0 propose nearby
translation corrections, then labels only a small deterministic subset of
*moved* native/proposal pairs with the exact CAD/DexNet evaluator.

The output is compact. It does not duplicate image features: training later
loads the matching pre-enhancer image map from the existing Rep-A cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from rep_a_common import (
    check_runtime_sources, choose, cdf_targets, digest, exclusive_run,
    file_sha, frame_identity, list_frames, perturb_depth, read_frame,
    save_json, save_npz, seed_for, training_case,
)
from rep_c2v2_common import class_from_delta, select_move_subset
from rep_followup_common import checked_manifest, load_scorer, memory_status, release_memory, source_digest
from rep_fullpath_runtime import expand_actions, load_stage1, parse_offsets
from infer_rep_fullpath import query_indices, score_queries


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--stage1-checkpoint", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--a1-checkpoint", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--camera", default="realsense")
    p.add_argument("--pose-depth-mode", default="global_film")
    p.add_argument("--offsets-mm", default="-40,-20,-10,0,10,20,40")
    p.add_argument("--query-limit", type=int, default=64,
                   help="Stage-1 queries scored by A1 before proposal subsampling")
    p.add_argument("--move-limit", type=int, default=16,
                   help="Maximum moved proposals exact-labelled per frame; 0 keeps all")
    p.add_argument("--score-query-chunk", type=int, default=64)
    p.add_argument("--max-bias-mm", type=float, default=20.)
    p.add_argument("--max-scale", type=float, default=.03)
    p.add_argument("--smooth-mm", type=float, default=10.)
    p.add_argument("--nominal-prob", type=float, default=.25)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--eval-chunk", type=int, default=128)
    p.add_argument("--fc-mode", choices=("official","reuse_contacts"), default="reuse_contacts")
    p.add_argument("--verify-n", type=int, default=0)
    p.add_argument("--min-host-free-gib", type=float, default=6.)
    p.add_argument("--feature-atol", type=float, default=5e-4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--repair-corrupt", action="store_true")
    return p


def main():
    args = parser().parse_args()
    sys.argv = [sys.argv[0]]
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid sharding")
    if args.query_limit < 1 or args.move_limit < 0 or args.score_query_chunk < 1:
        raise ValueError("Invalid query/move/chunk limits")
    if not 0 <= args.nominal_prob <= 1:
        raise ValueError("nominal-prob must be in [0,1]")

    manifest = check_runtime_sources(args.cache_root)
    if file_sha(args.stage1_checkpoint) != manifest["stage1_sha256"]:
        raise RuntimeError("Use the same Stage-1 checkpoint as Rep-A")
    if manifest["camera"] != args.camera or manifest["pose_depth_mode"] != args.pose_depth_mode:
        raise RuntimeError("Camera/pose-depth-mode differs from Rep-A cache")

    offsets = parse_offsets(args.offsets_mm)
    a1, a1_meta = load_scorer(args.a1_checkpoint, args.cache_root, torch.device(args.device))
    if a1_meta["variant"] != "A1":
        raise ValueError("Rep-C2-v2 requires the A1 checkpoint")
    a1.requires_grad_(False)

    paths = list_frames(args.cache_root, "train")
    scenes = sorted({frame_identity(p)[0] for p in paths})
    owned = set(scenes[args.shard_id::args.num_shards])
    paths = [p for p in paths if frame_identity(p)[0] in owned]
    if args.max_frames > 0:
        paths = paths[:args.max_frames]

    protocol = {
        "version": 1,
        "experiment": "Rep-C2-v2-train-cache",
        "cache_contract": a1_meta["contract"],
        "stage1_sha": file_sha(args.stage1_checkpoint),
        "a1_sha": file_sha(args.a1_checkpoint),
        "camera": args.camera,
        "pose_depth_mode": args.pose_depth_mode,
        "offsets_mm": offsets.tolist(),
        "query_limit": args.query_limit,
        "move_limit": args.move_limit,
        "training_error_distribution": {
            "max_bias_mm": args.max_bias_mm,
            "max_scale": args.max_scale,
            "smooth_mm": args.smooth_mm,
            "nominal_prob": args.nominal_prob,
        },
        "proposal": "A1 fixed_0 on regenerated full-path candidates under JOINT depth error",
        "labels": "fresh exact CAD/DexNet utility for native + proposed physical action only",
        "proposal_subset": "half highest A1 margin + half uniform over remaining moved queries",
        "image_evidence": "NOT stored here; loaded from matching Rep-A pre-enhancer cache at training time",
        "seed": args.seed,
        "fc_mode": args.fc_mode,
        "verify_n": args.verify_n,
        "code": source_digest((
            "prepare_rep_c2v2_train.py","rep_c2v2_common.py","rep_fullpath_runtime.py",
            "infer_rep_fullpath.py","exact_action_graspnet_evaluator.py",
        )),
    }
    signature = digest(protocol)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_run(root/".protocol.lock", wait=True):
        checked_manifest(root, protocol)

    # Late heavy imports after CLI isolation.
    from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
    from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
    from models.economicgrasp_bip3d import pred_decode_center_view_angle

    device = torch.device(args.device)
    stage, replay, use_fuse = load_stage1(args.stage1_checkpoint, device, args.pose_depth_mode)
    dataset = GraspNetMultiDataset(
        args.dataset_root, split="train", camera=args.camera, num_points=20000,
        remove_outlier=True, augment=False, load_label=False, use_gt_depth=False,
        use_fuse_depth=use_fuse, min_depth=.2, max_depth=1., bin_num=256,
    )
    index = {(int(str(s).split("_")[-1]), i % 256): i for i,s in enumerate(dataset.scene_list())}
    evaluator = ExactGraspNetActionEvaluator(
        args.dataset_root, args.camera, split="train",
        collision_chunk=args.eval_chunk, fc_mode=args.fc_mode,
        verify_n=args.verify_n, strict=True,
    )

    current_scene = None
    completed = skipped_empty = examples = 0
    class_counts = np.zeros(3, np.int64)

    try:
        for src in paths:
            sid, aid = frame_identity(src)
            out = root/"train"/f"scene_{sid:04d}"/f"ann_{aid:04d}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            with exclusive_run(out.with_suffix(".lock")):
                if out.exists() and args.resume:
                    try:
                        with np.load(out, allow_pickle=False) as z:
                            if str(z["signature"]) != signature:
                                raise RuntimeError(f"Changed Rep-C2-v2 mining contract: {out}")
                            target = np.asarray(z["target_class"], np.int64)
                        class_counts += np.bincount(target, minlength=3)
                        examples += len(target)
                        continue
                    except Exception:
                        if not args.repair_corrupt:
                            raise
                elif out.exists():
                    raise FileExistsError(out)

                if sid != current_scene:
                    evaluator.scene_cache.clear()
                    release_memory()
                    current_scene = sid
                memory_status(args.min_host_free_gib)

                cached = read_frame(src, a1_meta["contract"])
                item = dataset[index[(sid,aid)]]
                allowed = ("img","K","camera_pose_vec","camera_gravity_vec","scene_idx","anno_idx","token_valid_mask")
                batch = collate_fn([{k:item[k] for k in allowed if k in item}])
                del item
                batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
                batch.update(
                    cva_export_angle_feature=False,
                    cva_compute_diagnostics=False,
                    geometry_compute_diagnostics=False,
                )

                ep0 = replay.capture(batch)
                depth0 = replay.nominal_depth
                # Verify the independent RGB evidence used later is exactly the
                # same pre-enhancer map as the existing Rep-A cache, modulo its
                # recorded float16 quantization.
                feat = replay.image_feature
                feat_q = feat.to(
                    torch.float16 if manifest["feature_dtype"]=="float16" else torch.float32
                ).float().cpu().numpy()
                feature_max_abs = float(np.max(np.abs(feat_q-cached["image_feature"])))
                if feature_max_abs > args.feature_atol:
                    raise RuntimeError(
                        f"Rep-A RGB cache mismatch {sid}/{aid}: {feature_max_abs} > {args.feature_atol}"
                    )

                s = seed_for(args.seed, "rep-c2v2-train-case", sid, aid)
                case = training_case(
                    s, args.max_bias_mm, args.max_scale,
                    args.smooth_mm, args.nominal_prob,
                )
                pert, err = perturb_depth(depth0[0], case, s)
                ep = replay.run(batch, pert[None], pert[None])
                native_all = pred_decode_center_view_angle(ep, use_cdf=True)[0].cpu().numpy()
                ids = query_indices(native_all, args.query_limit)
                native = native_all[ids]
                actions, valid, zero = expand_actions(native, offsets)

                h,w = pert.shape[-2:]
                tokens = ep["kview_base_token_sel_idx"][0][torch.as_tensor(ids, device=device)]
                model_data = {
                    "image_feature": feat_q,
                    "depth": pert,
                    "K": batch["K"][0],
                    "objectness": ep["objectness_score"][0].reshape(2,h,w),
                    "graspness": ep["graspness_score"][0].reshape(1,h,w),
                    "actions": torch.from_numpy(actions).to(device),
                    "valid": torch.from_numpy(valid).to(device),
                    "offsets_mm": torch.from_numpy(offsets).to(device),
                    "zero_index": zero,
                    "token_ids": tokens,
                }
                # image_feature must be torch for A1, and preserve exact cached quantization.
                model_data["image_feature"] = torch.from_numpy(feat_q).to(device)
                prob = score_queries(a1, model_data, args.score_query_chunk)
                pu = prob.mean(-1)
                proposal = choose(pu, valid, zero, 0.)
                q_all = np.arange(len(ids))
                moved = np.flatnonzero(proposal != zero)

                if moved.size:
                    margins = pu[proposal[moved], moved] - pu[zero, moved]
                    keep_local = select_move_subset(margins, args.move_limit)
                    qkeep = moved[keep_local]
                else:
                    qkeep = np.empty(0, np.int64)

                if qkeep.size:
                    native_g = actions[zero, qkeep]
                    prop_g = actions[proposal[qkeep], qkeep]
                    physical = np.concatenate((native_g, prop_g), axis=0)
                    result = evaluator.evaluate(sid, aid, physical)
                    fr = result.friction.reshape(2, len(qkeep))
                    y = cdf_targets(fr)
                    utility = y.mean(-1).astype(np.float32)
                    delta = (utility[1]-utility[0]).astype(np.float32)
                    target = class_from_delta(delta)
                    pair_actions = np.stack((native_g, prop_g), axis=0).astype(np.float32)
                    pair_prob = np.stack(
                        (prob[zero,qkeep], prob[proposal[qkeep],qkeep]), axis=0
                    ).astype(np.float32)
                    selected_offsets = offsets[proposal[qkeep]].astype(np.float32)
                    original_score = native[:,0][qkeep].astype(np.float32)
                    query_ids = ids[qkeep].astype(np.int64)
                    pure = result.pure_collision.reshape(2,len(qkeep))
                    empty = result.empty.reshape(2,len(qkeep))
                else:
                    fr = np.empty((2,0), np.float32)
                    utility = np.empty((2,0), np.float32)
                    delta = np.empty(0, np.float32)
                    target = np.empty(0, np.int64)
                    pair_actions = np.empty((2,0,17), np.float32)
                    pair_prob = np.empty((2,0,6), np.float32)
                    selected_offsets = np.empty(0, np.float32)
                    original_score = np.empty(0, np.float32)
                    query_ids = np.empty(0, np.int64)
                    pure = np.empty((2,0), bool)
                    empty = np.empty((2,0), bool)
                    skipped_empty += 1

                payload = {
                    "signature": np.array(signature),
                    "scene_id": np.array(sid),
                    "anno_id": np.array(aid),
                    "case": np.array(case),
                    "query_ids": query_ids,
                    "actions": pair_actions,
                    "probabilities": pair_prob,
                    "offsets_mm": selected_offsets,
                    "original_native_score": original_score,
                    "friction": fr,
                    "utility": utility,
                    "delta_utility": delta,
                    "target_class": target,
                    "pure_collision": pure,
                    "empty": empty,
                    "depth_rms_mm": np.array(err["depth_rms_mm"]),
                    "depth_bias_mm": np.array(err["depth_bias_mm"]),
                    "num_stage1_queries": np.array(len(native_all)),
                    "num_scored_queries": np.array(len(ids)),
                    "num_a1_moves_before_subset": np.array(len(moved)),
                    "feature_max_abs": np.array(feature_max_abs),
                }
                save_npz(out, payload)
                class_counts += np.bincount(target, minlength=3)
                examples += len(target)
                completed += 1

                del cached,batch,ep0,ep,native_all,native,actions,valid,model_data,prob
                replay.clear()
                release_memory()
                if completed % 20 == 0:
                    print(
                        f"[REP-C2-v2 MINE] frames={completed}/{len(paths)} "
                        f"examples={examples} class={class_counts.tolist()} {memory_status()}",
                        flush=True,
                    )
    finally:
        evaluator.scene_cache.clear()
        release_memory()

    save_json(root/f"mine_shard{args.shard_id}.json", {
        "signature": signature,
        "selected_frames": len(paths),
        "completed_new_frames": completed,
        "empty_move_frames": skipped_empty,
        "examples_seen_this_process_including_resume": examples,
        "class_counts_harm_equiv_benefit": class_counts.tolist(),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
    })
    print(
        f"[REP-C2-v2 MINE] done frames={len(paths)} examples={examples} "
        f"class={class_counts.tolist()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
