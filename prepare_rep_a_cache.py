#!/usr/bin/env python3
"""Export PRE-enhancer image maps for existing Rep-P0 fixed actions.

No re-mining, no CAD/DexNet, no observed/rendered depth input to the model.
Every native action is replay-checked against P0. Existing label files are read
only. One frame in memory, scene-level sharding, atomic writes and resume.
"""
from __future__ import annotations
import argparse
import dataclasses
import gc
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from rep_a_common import (VERSION, BASE_MAIN, SPLITS, array_sha, digest, file_sha,
                         fixed_data, frame_identity, list_frames, read_frame,
                         save_json, save_npz, save_torch, exclusive_run, load_torch)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--p0-cache-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--split", choices=SPLITS, default="train")
    p.add_argument("--camera", default="realsense")
    p.add_argument("--pose-depth-mode", default="global_film", choices=("global_film", "ray_gravity_film", "none"))
    p.add_argument("--feature-dtype", default="float16", choices=("float16", "float32"))
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--replay-atol", type=float, default=5e-5)
    p.add_argument("--repair-corrupt", action="store_true")
    p.add_argument("--min-host-free-gib", type=float, default=4.)
    p.add_argument("--progress-every", type=int, default=10)
    return p


def memory_guard(min_free):
    fields = {}
    try:
        fields = {a.split(':')[0]: int(a.split()[1]) for a in Path('/proc/meminfo').read_text().splitlines()}
        available = fields['MemAvailable']/1024**2
        rss = next(int(x.split()[1])/1024**2 for x in Path('/proc/self/status').read_text().splitlines()
                   if x.startswith('VmRSS:'))
        if available < min_free:
            raise MemoryError(f"Host MemAvailable={available:.2f} GiB; stop safely and resume later")
        return f"rss={rss:.2f}GiB available={available:.2f}GiB"
    except (FileNotFoundError, KeyError):
        return "memory_monitor_unavailable"


def main():
    args = args_parser().parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid scene shard")
    # utils.arguments parses sys.argv on import. Remove our already-parsed flags.
    sys.argv = [sys.argv[0]]
    from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
    from models.economicgrasp_dpt_distill import economicgrasp_dpt_student
    from models.economicgrasp_bip3d import pred_decode_center_view_angle
    from utils.arguments import cfgs

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    expected = {"version": VERSION, "base_main": BASE_MAIN,
                "stage1_sha256": file_sha(args.checkpoint), "camera": args.camera,
                "upstream_sha256": {rel: file_sha(Path(__file__).resolve().parent/rel) for rel in
                    ("models/grasp_spatial_enhancer.py", "models/kview_query_transformer.py", "models/economicgrasp_bip3d.py")},
                "pose_depth_mode": args.pose_depth_mode, "feature_dtype": args.feature_dtype,
                "evidence": "proposal_head.output[0], before spatial_enhancer; no spatial downsampling",
                "intervention": "reader_depth_only; actions/labels unchanged"}
    contract = digest(expected)
    with exclusive_run(root/".manifest.lock", wait=True):
        if (root/"manifest.json").exists():
            if json.loads((root/"manifest.json").read_text()) != expected:
                raise RuntimeError("Different cache contract. Use a NEW output-root; do not mix exports.")
        else:
            save_json(root/"manifest.json", expected)

    paths = list_frames(args.p0_cache_root, args.split)
    scenes = sorted({frame_identity(p)[0] for p in paths})
    owned = {s for i, s in enumerate(scenes) if i % args.num_shards == args.shard_id}
    paths = [p for p in paths if frame_identity(p)[0] in owned]
    if args.max_frames > 0:
        paths = paths[:args.max_frames]
    pending, resumed = [], 0
    # Resume BEFORE constructing dataset/model or loading CAD/depth assets.
    for src in paths:
        dst = root/args.split/src.parent.name/src.name
        source_hash = file_sha(src)
        if dst.exists():
            try:
                saved = read_frame(dst)  # corruption is repairable; protocol mismatch is NOT
            except Exception:
                if not args.repair_corrupt:
                    raise
                saved = None
            if saved is not None:
                if str(saved['contract']) != contract or str(saved['p0_sha256']) != source_hash:
                    raise RuntimeError(f"Protocol or fixed-label input changed at {dst}; use a new output-root")
                resumed += 1
                del saved
                continue
        pending.append((src, dst, source_hash))
    print(f"[REP-A PREP] scene-shard={args.shard_id}/{args.num_shards} pending={len(pending)} resumed={resumed}", flush=True)
    if not pending:
        return
    memory_guard(args.min_host_free_gib)
    cfgs.use_top4_view_infer = False
    cfgs.kview_mode, cfgs.kview_k, cfgs.use_cdf, cfgs.use_obs_depth = "A1", 1, True, False
    cfgs.pose_depth_mode = args.pose_depth_mode
    ckpt = load_torch(args.checkpoint)
    if ckpt.get('geometry_depth_source', 'pred') != 'pred' or ckpt.get('use_obs_depth', False):
        raise ValueError("Requires the RGB-only Stage-1 checkpoint")
    if ckpt.get('pose_depth_mode', args.pose_depth_mode) != args.pose_depth_mode:
        raise ValueError("Checkpoint/POSE_DEPTH_MODE mismatch")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = economicgrasp_dpt_student(is_training=False, use_obs_depth=False, use_cdf=True,
                                      pose_depth_mode=args.pose_depth_mode, vis_dir=None).to(device)
    state = ckpt['model_state_dict']
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    optional = 'rgb_geometry_diagnostics.'
    if any(not k.startswith(optional) for k in result.missing_keys+result.unexpected_keys):
        raise RuntimeError(f"Strict Stage-1 load failed: {result}")
    use_fuse = bool(ckpt.get('use_fuse_depth', False))
    del state, ckpt
    gc.collect()
    model.eval().requires_grad_(False)
    group = model.kview_grasp_module.group
    group_config = dataclasses.asdict(model.kview_config)
    group_config.update(vis_dir=None, save_npz=False)
    enh = model.spatial_enhancer
    init = {"contract": contract, "model_config": {
        "channels": group.feat_dim, "out_dim": group.out_dim, "group_config": group_config,
        "enhancer": {k: getattr(enh, k) for k in ('embed_dims', 'feature_3d_dim', 'min_depth',
                        'max_depth', 'num_depth', 'detach_depth_grad', 'use_post_norm', 'prob_eps')}},
        "enhancer_state": {k: v.detach().cpu() for k, v in enh.state_dict().items()},
        "group_state": {k: v.detach().cpu() for k, v in group.state_dict().items()}}
    with exclusive_run(root/".manifest.lock", wait=True):
        if (root/"reader_init.pt").exists():
            old = load_torch(root/"reader_init.pt")
            if old['contract'] != contract or old['model_config'] != init['model_config']:
                raise RuntimeError("Reader initialization changed")
            for part in ('enhancer_state', 'group_state'):
                if old[part].keys() != init[part].keys() or any(not torch.equal(v, old[part][k]) for k,v in init[part].items()):
                    raise RuntimeError("Reader weights changed")
            del old
        else:
            save_torch(root/"reader_init.pt", init)
    del init

    dataset = GraspNetMultiDataset(args.dataset_root, split=args.split, camera=args.camera,
        num_points=20000, remove_outlier=True, augment=False, load_label=False,
        use_gt_depth=False, use_fuse_depth=use_fuse, min_depth=.2, max_depth=1., bin_num=256)
    mapping = {(int(str(scene).split('_')[-1]), idx % 256): idx
               for idx, scene in enumerate(dataset.scene_list())}
    captured = {}
    def capture(_module, _inputs, output):
        captured['raw'] = output[0].detach()
    hook = model.proposal_head.register_forward_hook(capture)
    try:
        for i, (src, dst, source_hash) in enumerate(pending):
            with exclusive_run(dst.with_suffix('.lock')):
                d = fixed_data(src)
                sid, aid = frame_identity(src)
                item = dataset[mapping[(sid, aid)]]
                # Model gets only RGB/calibration. Dataset preprocessing is kept identical
                # to P0, which includes the repository's workspace crop convention.
                allowed = ('img', 'K', 'camera_pose_vec', 'camera_gravity_vec',
                           'scene_idx', 'anno_idx', 'token_valid_mask')
                batch = collate_fn([{k: item[k] for k in allowed if k in item}])
                del item
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
                batch.update(cva_export_angle_feature=False, cva_compute_diagnostics=False,
                             geometry_compute_diagnostics=False)
                with torch.inference_mode():
                    ep = model(batch)
                    native = pred_decode_center_view_angle(ep, use_cdf=True)[0]
                ids = torch.from_numpy(d['query_ids'].astype(np.int64)).to(device)
                replay = native.index_select(0, ids).cpu().numpy()
                err = float(np.max(np.abs(replay-d['actions'][int(d['zero_index'])])))
                if err > args.replay_atol:
                    raise RuntimeError(f"Native replay mismatch {src}: max_abs={err}. Wrong checkpoint/protocol; labels NOT reused.")
                raw = captured.pop('raw')[0].cpu().numpy().astype(args.feature_dtype)
                depth = ep['depth_map_used_for_geometry'][0].cpu().numpy().astype(np.float32)
                if not np.isfinite(raw).all() or not np.isfinite(depth).all():
                    raise FloatingPointError('Non-finite exported image/depth; use float32 if fp16 overflowed')
                if depth.ndim == 2:
                    depth = depth[None]
                h, w = depth.shape[-2:]
                payload = {**d, 'version': np.array(VERSION), 'contract': np.array(contract),
                    'p0_sha256': np.array(source_hash),
                    'action_sha': np.array(array_sha(d['actions'], d['valid'], d['friction'], d['query_ids'])),
                    'image_feature': raw, 'depth': depth, 'K': batch['K'][0].cpu().numpy(),
                    'objectness': ep['objectness_score'][0].reshape(2,h,w).cpu().numpy().astype(np.float32),
                    'graspness': ep['graspness_score'][0].reshape(1,h,w).cpu().numpy().astype(np.float32),
                    'token_ids': ep['kview_base_token_sel_idx'][0].index_select(0,ids).cpu().numpy(),
                    'replay_max_abs': np.array(err)}
                if shutil.disk_usage(root).free < max(2*1024**3, 2*sum(v.nbytes for v in payload.values())):
                    raise OSError("Insufficient disk space; partial completed cache is resumable")
                save_npz(dst, payload)
                if i == 0:
                    print(f"[REP-A PREP] raw_feature={raw.shape} stored={dst.stat().st_size/1024**2:.2f}MiB/frame; "
                          f"remaining shard estimate={(len(pending)-1)*dst.stat().st_size/1024**3:.1f}GiB", flush=True)
                del ep, batch, native, ids, replay, raw, depth, payload, d
            if (i+1) % max(args.progress_every, 1) == 0:
                gc.collect()
                print(f"[REP-A PREP] {i+1}/{len(pending)} {memory_guard(args.min_host_free_gib)}", flush=True)
    finally:
        hook.remove()
    save_json(root/args.split/f'prepare_shard_{args.shard_id:02d}.json',
              {'contract': contract, 'processed': len(pending), 'resumed': resumed,
               'num_shards': args.num_shards, 'shard_id': args.shard_id})


if __name__ == '__main__':
    main()
