#!/usr/bin/env python3
"""Visualize EconomicGrasp-DPT-CVA-CDF / DCR behavior on selected GraspNet frames.

Default coverage: RealSense test_seen/test_similar/test_novel, every scene, frames
0/128/255. The script is diagnostic only: no GT/CAD/DexNet signal is fed to the
network. Sensor depth is shown only as optional visual context.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

from dcr_cva_common import DCR_VERSION, make_outputs
from e1e2_common import (VERSION, file_sha, get_batch, load_torch, make_dataset,
                         perturb_depth, seed_all, seed_for)
from tools.grasp_behavior_visualizer import (
    BehaviorVisConfig, BehaviorVisualizer, backproject_depth, denormalize_rgb,
    save_cdf_heatmap, save_center_shift_overlay, save_depth_panel,
    save_evaluation_overlay, save_evaluator_friction_overlay, save_feature_panel, save_grasp_overlay,
    save_local_patch_overlay, save_offset_response, save_ply,
    save_pointcloud_views, save_proposal_panel, save_rank_residual_panel,
    to_numpy,
)


SPLIT_SCENES = {
    'test_seen': range(100, 130),
    'test_similar': range(130, 160),
    'test_novel': range(160, 190),
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root', 'stage1-checkpoint', 'dcr-checkpoint',
                 'output-root'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--air-checkpoint', default='')
    p.add_argument('--official-root', default='',
                   help='Optional DCR inference root containing official accuracy.npy')
    p.add_argument('--run-evaluator', action='store_true',
                   help='Run exact GraspNet evaluator on these sparse selected frames')
    p.add_argument('--evaluator-methods', default='stage1,native',
                   help='DCR output modes to send through sparse-frame evaluator')
    p.add_argument('--splits',
                   default='test_seen,test_similar,test_novel')
    p.add_argument('--frames', default='0,128,255')
    p.add_argument('--scenes', default='',
                   help='Optional comma-separated scene ids; empty = every scene in split')
    p.add_argument('--cases',
                   default='nominal,bias:-25,bias:25,scale:-0.03,scale:0.03,smooth:10')
    p.add_argument('--items', default='all',
                   help='all or comma-separated: rgb,depth,pointcloud,features,proposal,'
                        'stage1,dcr,cdf,local,grasp_delta,evaluation,air')
    p.add_argument('--topk', type=int, default=20)
    p.add_argument('--max-points', type=int, default=50000)
    p.add_argument('--query-limit', type=int, default=0,
                   help='0 = all Stage-1 queries; use e.g. 256 for faster visualization')
    p.add_argument('--query-chunk', type=int, default=64)
    p.add_argument('--rank-strength', type=float, default=0.,
                   help='DCR display defaults to frozen Stage-1 ranking')
    p.add_argument('--shard-id', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--overwrite', action='store_true')
    return p


def _parse_ints(text):
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def _parse_strs(text):
    return [x.strip() for x in text.split(',') if x.strip()]


def _load_model(args, device):
    ck = load_torch(args.dcr_checkpoint)
    if ck['version'] not in (VERSION, DCR_VERSION):
        raise RuntimeError('Unsupported DCR/E1 checkpoint')
    protocol, cfg = ck['protocol'], ck['protocol']['config']
    if protocol['reference_sha256'] != file_sha(args.stage1_checkpoint):
        raise RuntimeError('Stage-1/DCR checkpoint mismatch')
    from models.economicgrasp_cva_centers import load_reference
    from models.economicgrasp_cva_dcr import DecoupledCenterRankingCVA
    model = DecoupledCenterRankingCVA(
        load_reference(args.stage1_checkpoint, device),
        protocol['offsets_mm'],
        cfg['group_chunk'],
        cfg.get('rank_hidden', 128),
        cfg.get('rank_bound', .5),
        cfg['seed'],
    ).to(device)
    if ck['version'] == DCR_VERSION:
        model.load_learned_state(ck['model'])
    else:
        model.warm_start_corrector(ck['model'])
    model.eval()
    del ck
    return model, protocol, cfg


def _load_air(args, device):
    if not args.air_checkpoint:
        return None
    from dcr_air_common import AIR_VERSION
    from models.economicgrasp_cva_air import ActionImageDCR, load_frozen_dcr
    ck = load_torch(args.air_checkpoint)
    if ck.get('version') != AIR_VERSION:
        raise RuntimeError('Unsupported AIR checkpoint')
    p, cfg = ck['protocol'], ck['protocol']['config']
    if p['base_dcr_sha256'] != file_sha(args.dcr_checkpoint):
        raise RuntimeError('AIR/DCR checkpoint mismatch')
    base, _ = load_frozen_dcr(
        args.stage1_checkpoint, args.dcr_checkpoint, device)
    air = ActionImageDCR(
        base, cfg['hidden'], cfg['residual_bound'], cfg['seed']).to(device)
    air.load_learned_state(ck['model'])
    air.eval()
    del ck
    return air


def _score_corrector_inspect(model, batch, pack, case, case_seed,
                             query_limit, query_chunk):
    """Reproduce DCR corrector forward while retaining visualization tensors."""
    from models.economicgrasp_cva_centers import reference_candidates
    bundle, active, stage1_ep = reference_candidates(
        model.reference, batch, pack, case, case_seed,
        model.corrector.offsets_mm.cpu().numpy(), query_limit,
        return_end_points=True)
    feature, proposal_logits, raw, enhanced, spatial_aux = (
        model.corrector.encode_image(
            batch, pack, active, return_maps=True))

    n = bundle['actions'].shape[1]
    logits_list, latent_list = [], []
    local_debug = {}
    for start in range(0, n, query_chunk):
        end = min(n, start + query_chunk)
        sub = {
            k: (v[:, start:end] if k in ('actions', 'valid') else
                v[start:end] if k in
                ('token_ids', 'view_xyz', 'angle_ids', 'depth_ids')
                else v)
            for k, v in bundle.items()
        }
        chunk_debug = {}
        result = model.corrector.score_bundle(
            feature, proposal_logits, batch, active, sub,
            return_features=True, debug_sink=chunk_debug)
        logit, latent = result
        logits_list.append(logit)
        latent_list.append(latent)
        # The first chunk contains deterministic high-coverage debug arrays.
        if start == 0:
            local_debug = chunk_debug
    logits = torch.cat(logits_list, 1)
    latent = torch.cat(latent_list, 1)
    native_score = bundle['actions'][model.zero, :, 0]
    residual = model.ranker(
        latent, logits, native_score,
        model.corrector.offsets_mm, model.zero)
    outputs, selected = make_outputs(
        logits, residual, bundle, model.zero, 0.)
    return {
        'bundle': bundle,
        'active_depth': active,
        'stage1_ep': stage1_ep,
        'raw_feature': raw,
        'enhanced_feature': enhanced,
        'feature_full': feature,
        'proposal_logits': proposal_logits,
        'spatial_aux': spatial_aux,
        'logits': logits,
        'latent': latent,
        'rank_residual': residual,
        'outputs': outputs,
        'selected': selected,
        'local_debug': local_debug,
    }


def _sensor_context(ds, lookup, sid, aid):
    idx = lookup[(sid, aid)]
    rgb = np.asarray(Image.open(ds.colorpath[idx]).convert('RGB'),
                     dtype=np.float32) / 255.
    # Model works on crop+resize. For direct overlays use the model tensor RGB.
    # Sensor depth below is taken from dataset item so it is in the same 448 grid.
    item = ds[idx]
    sensor = np.asarray(item.get('sensor_depth_m'), np.float32)
    return rgb, sensor


def _scalar_endpoint_summary(*dicts):
    out = {}
    for prefix, d in dicts:
        for key, val in d.items():
            if torch.is_tensor(val) and val.numel() == 1:
                out[f'{prefix}/{key}'] = float(val.detach().cpu())
            elif isinstance(val, (float, int, np.floating, np.integer)):
                out[f'{prefix}/{key}'] = float(val)
    return out


def _stage1_response_maps(ep, bundle, image_hw):
    """Project selected-token ViewNet confidence/entropy back to the image grid."""
    h, w = image_hw
    maps = {}
    vs = ep.get('view_score')
    token = bundle['token_ids']
    qids = bundle['query_ids']
    if torch.is_tensor(vs) and vs.ndim == 3:
        v = vs[0][qids]
        prob = torch.softmax(v.float(), -1)
        conf = prob.max(-1).values
        entropy = -(prob * prob.clamp_min(1e-9).log()).sum(-1)
        for name, values in (('view_top_prob', conf), ('view_entropy', entropy)):
            img = torch.full((h * w,), float('nan'), device=values.device)
            img[token.long()] = values
            maps[name] = img.reshape(h, w).detach().cpu().numpy()
    return maps


def _save_stage1_maps(out, rgb, maps):
    if not maps:
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 1 + len(maps),
                             figsize=(5 * (1 + len(maps)), 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(rgb); axes[0].set_title('RGB'); axes[0].axis('off')
    for ax, (name, arr) in zip(axes[1:], maps.items()):
        im = ax.imshow(arr, cmap='viridis')
        ax.set_title(name); ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _save_spatial_aux_maps(out, spatial_aux):
    candidates = []
    for key, val in spatial_aux.items():
        if not torch.is_tensor(val):
            continue
        x = val.detach().cpu()
        while x.ndim > 2 and x.shape[0] == 1:
            x = x[0]
        if x.ndim == 2:
            candidates.append((key, x.numpy()))
        elif x.ndim == 3 and x.shape[0] <= 4:
            for c in range(x.shape[0]):
                candidates.append((f'{key}[{c}]', x[c].numpy()))
    if not candidates:
        return
    import matplotlib.pyplot as plt
    n = min(len(candidates), 12)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, (name, arr) in zip(axes, candidates[:n]):
        im = ax.imshow(arr, cmap='viridis')
        ax.set_title(name, fontsize=8); ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    for ax in axes[n:]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _match_query_actions(nominal, current, nominal_actions, current_actions):
    nq = to_numpy(nominal['bundle']['query_ids'], np.int64)
    cq = to_numpy(current['bundle']['query_ids'], np.int64)
    lookup = {int(q): i for i, q in enumerate(cq)}
    ni, ci = [], []
    for i, q in enumerate(nq):
        j = lookup.get(int(q))
        if j is not None:
            ni.append(i); ci.append(j)
    if not ni:
        return None, None
    return nominal_actions[np.asarray(ni)], current_actions[np.asarray(ci)]


def _load_official_accuracy(root, method, case, split, sid, aid):
    """Load evaluator rank x threshold outcomes when this exact frame was evaluated."""
    if not root:
        return None
    root = Path(root)
    protocol_path = root / 'protocol.json'
    if not protocol_path.is_file():
        return None
    p = json.loads(protocol_path.read_text())
    from e1e2_common import case_key, schedule
    frames = schedule(split, p['sample_interval'])
    try:
        global_idx = frames.index((sid, aid))
    except ValueError:
        return None
    # accuracy layout is [30 scenes, frames-per-scene, top50, thresholds].
    scene_ids = list(SPLIT_SCENES[split])
    si = scene_ids.index(sid)
    per_scene = len(frames) // len(scene_ids)
    fi = global_idx - si * per_scene
    path = root / 'official' / method / case_key(case) / split / 'accuracy.npy'
    if not path.is_file():
        return None
    arr = np.load(path, allow_pickle=False)
    if arr.ndim < 4 or si >= arr.shape[0] or fi >= arr.shape[1]:
        return None
    return arr[si, fi]


def _save_case(vis, out, rgb_model, sensor, K, pack, result, case, args,
               eval_root='', air_result=None, frame_evaluator=None,
               evaluator_methods=()):
    bundle = result['bundle']
    outputs = result['outputs']
    score = outputs['stage1'][:, 0]
    predicted = pack[0]

    if vis.wants('rgb'):
        from tools.grasp_behavior_visualizer import save_rgb
        save_rgb(out / '00_rgb_model_input.png', rgb_model,
                 title='Model RGB input (448x448)')

    if vis.wants('depth'):
        save_depth_panel(
            out / '01_depth_geometry.png', predicted,
            result['active_depth'], title=case, sensor=sensor)

    if vis.wants('pointcloud'):
        pred_pts, _ = backproject_depth(predicted, K, stride=2)
        active_pts, _ = backproject_depth(result['active_depth'], K, stride=2)
        save_pointcloud_views(
            out / '02_pointcloud_geometry.png',
            {'predicted RGB depth': pred_pts,
             f'active geometry ({case})': active_pts},
            max_points=args.max_points, title=f'RGB-derived geometry: {case}')
        save_ply(out / '02_predicted_depth_cloud.ply', pred_pts)
        save_ply(out / '02_active_depth_cloud.ply', active_pts)

    if vis.wants('features'):
        save_feature_panel(
            out / '03_feature_pre_post_enhancer.png',
            result['raw_feature'], result['enhanced_feature'],
            title=f'Image feature response: {case}')
        _save_spatial_aux_maps(
            out / '03b_spatial_enhancer_aux.png',
            result['spatial_aux'])

    if vis.wants('proposal'):
        save_proposal_panel(
            out / '04_proposal_maps.png', result['proposal_logits'],
            rgb=rgb_model, title=f'Objectness / graspness: {case}')

    if vis.wants('stage1'):
        save_grasp_overlay(
            out / '05_stage1_native_grasps.png', rgb_model, K,
            {'Stage-1 native': to_numpy(outputs['native'])},
            topk=args.topk, title=f'Stage-1 native grasps: {case}')
        maps = _stage1_response_maps(
            result['stage1_ep'], bundle, rgb_model.shape[:2])
        _save_stage1_maps(out / '05b_view_response_maps.png', rgb_model, maps)

    if vis.wants('dcr'):
        save_grasp_overlay(
            out / '06_dcr_selected_grasps.png', rgb_model, K,
            {'native': to_numpy(outputs['native']),
             'DCR center-corrected': to_numpy(outputs['stage1'])},
            topk=args.topk, title=f'DCR physical correction: {case}')
        save_center_shift_overlay(
            out / '06b_dcr_center_shift.png', rgb_model, K,
            to_numpy(outputs['native']), to_numpy(outputs['stage1']),
            scores=to_numpy(score), topk=max(args.topk, 50),
            title=f'Native -> DCR center shift: {case}')
        save_offset_response(
            out / '06c_dcr_selected_offset_map.png', rgb_model, K,
            bundle, result['selected'],
            result_model_offsets(bundle, result), score,
            max_queries=max(args.topk, 100),
            title=f'DCR selected camera-z offsets: {case}')
        save_rank_residual_panel(
            out / '06d_rank_residual.png', result['rank_residual'],
            result_model_offsets(bundle, result), score,
            title=f'DCR rank-head residual (diagnostic): {case}')

    if vis.wants('cdf'):
        save_cdf_heatmap(
            out / '07_center_cdf_utility.png', result['logits'],
            result_model_offsets(bundle, result), score,
            max_queries=max(args.topk, 32),
            title=f'Center-hypothesis CDF utility: {case}')

    if vis.wants('local'):
        dbg = result['local_debug']
        if 'kview_debug_patch_uv0' in dbg:
            save_local_patch_overlay(
                out / '08_local_analysis_regions.png', rgb_model,
                dbg['kview_debug_patch_uv0'],
                dbg.get('kview_debug_patch_attn0'),
                dbg.get('kview_debug_center_uv0'),
                max_queries=min(args.topk, 16),
                title=f'CVA view-conditioned local regions: {case}')

    if vis.wants('evaluation'):
        # Save exactly what is sent to evaluator for every requested score policy.
        for method in evaluator_methods or ('stage1',):
            if method not in outputs:
                continue
            eval_grasps = to_numpy(outputs[method])
            np.save(
                out / f'09_eval_input_{method}.npy',
                eval_grasps, allow_pickle=False)
            if frame_evaluator is not None:
                er = frame_evaluator.evaluate(
                    result['_scene_id'], result['_anno_id'], eval_grasps)
                np.savez_compressed(
                    out / f'09_eval_result_{method}.npz', **er.as_npz())
                save_evaluator_friction_overlay(
                    out / f'09_post_evaluator_{method}.png',
                    rgb_model, K, er.grasps, er.friction_scores,
                    er.collision, topk=50,
                    title=f'Post-evaluator grasps ({method}): {case}')
                save_evaluation_overlay(
                    out / f'09_post_evaluator_prefix_accuracy_{method}.png',
                    rgb_model, K, er.grasps, er.accuracy,
                    topk=50,
                    title=f'Evaluator prefix accuracy ({method}): {case}')
            else:
                acc = _load_official_accuracy(
                    eval_root, method, case,
                    result['_split'], result['_scene_id'], result['_anno_id'])
                if acc is not None:
                    save_evaluation_overlay(
                        out / f'09_post_evaluator_prefix_accuracy_{method}.png',
                        rgb_model, K, eval_grasps, acc, topk=50,
                        title=f'Official evaluator outcome ({method}): {case}')
        if frame_evaluator is None and not eval_root:
            (out / '09_post_evaluator_unavailable.txt').write_text(
                'Exact post-evaluator visualization was not requested. '
                'Re-run with --run-evaluator for sparse frames 0/128/255, '
                'or provide --official-root for already-evaluated frames.\n')

    if vis.wants('air') and air_result is not None:
        fused, base, residual, air_bundle, _, diag = air_result
        from dcr_air_common import make_air_outputs
        air_outputs, dcr_sel, air_sel = make_air_outputs(
            base, fused, air_bundle, result['_model'].zero)
        save_grasp_overlay(
            out / '10_air_vs_dcr_grasps.png', rgb_model, K,
            {'DCR': to_numpy(air_outputs['dcr_stage1']),
             'AIR': to_numpy(air_outputs['air_stage1'])},
            topk=args.topk, title=f'AIR vs DCR: {case}')
        save_rank_residual_panel(
            out / '10b_air_logit_residual.png', residual,
            result_model_offsets(bundle, result), score,
            title=f'AIR evidence residual: {case}')
        np.savez_compressed(
            out / '10c_air_diagnostics.npz',
            residual=to_numpy(residual),
            visible_ratio=to_numpy(diag['visible_ratio']),
            attention_max=to_numpy(diag['attention_max']),
            dcr_selected=to_numpy(dcr_sel),
            air_selected=to_numpy(air_sel))

    if vis.cfg.save_npz:
        np.savez_compressed(
            out / 'behavior_tensors.npz',
            predicted_depth=to_numpy(predicted),
            active_depth=to_numpy(result['active_depth']),
            query_ids=to_numpy(bundle['query_ids']),
            token_ids=to_numpy(bundle['token_ids']),
            actions=to_numpy(bundle['actions']),
            valid=to_numpy(bundle['valid']),
            cdf_logits=to_numpy(result['logits']),
            selected=to_numpy(result['selected']),
            rank_residual=to_numpy(result['rank_residual']),
            stage1_score=to_numpy(score))

    scalars = _scalar_endpoint_summary(
        ('stage1', result['stage1_ep']),
        ('local', result['local_debug']),
        ('spatial', result['spatial_aux']))
    (out / 'scalar_diagnostics.json').write_text(
        json.dumps(scalars, indent=2, sort_keys=True))


def result_model_offsets(bundle, result):
    # Bundle keeps no offset tensor; infer from candidate camera-z relative to native.
    actions = bundle['actions']
    zero = int(bundle.get('zero', result['_model'].zero))
    native_z = actions[zero, :, 15]
    dz = ((actions[:, :, 15] - native_z[None]) * 1000.).median(1).values
    return dz


def main():
    args = parser().parse_args()
    sys.argv = [sys.argv[0]]
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError('Invalid shard id/count')
    if args.topk < 1 or args.query_chunk < 1 or args.query_limit < 0:
        raise ValueError('Invalid topk/query controls')
    device = torch.device(args.device)
    model, protocol, cfg = _load_model(args, device)
    air = _load_air(args, device)
    seed_all(cfg['seed'])

    splits = _parse_strs(args.splits)
    frames = _parse_ints(args.frames)
    selected_scenes = set(_parse_ints(args.scenes))
    cases = list(dict.fromkeys(['nominal'] + _parse_strs(args.cases)))
    for split in splits:
        if split not in SPLIT_SCENES:
            raise ValueError(split)
    for aid in frames:
        if not 0 <= aid < 256:
            raise ValueError(f'Invalid frame {aid}')

    vis_cfg = BehaviorVisConfig.from_strings(
        args.output_root, args.items, every=1, topk=args.topk,
        max_points=args.max_points)
    vis = BehaviorVisualizer(vis_cfg)
    evaluator_methods = tuple(_parse_strs(args.evaluator_methods))
    allowed_eval_methods = {'native', 'local', 'stage1', 'anchored'}
    if set(evaluator_methods) - allowed_eval_methods:
        raise ValueError(f'Bad evaluator methods: {set(evaluator_methods)-allowed_eval_methods}')
    frame_evaluator = None
    if args.run_evaluator and vis.wants('evaluation'):
        from tools.graspnet_frame_evaluator import SelectedFrameGraspEvaluator
        frame_evaluator = SelectedFrameGraspEvaluator(
            args.dataset_root, protocol['camera'], top_k=50)
    scene_jobs = []
    for split in splits:
        for sid in SPLIT_SCENES[split]:
            if selected_scenes and sid not in selected_scenes:
                continue
            scene_jobs.append((split, sid))
    scene_jobs = scene_jobs[args.shard_id::args.num_shards]

    run = {
        'branch_experiment': 'DCR behavior visualization',
        'camera': protocol['camera'],
        'splits': splits,
        'frames': frames,
        'cases': cases,
        'items': list(vis_cfg.items),
        'query_limit': args.query_limit,
        'query_chunk': args.query_chunk,
        'dcr_checkpoint': args.dcr_checkpoint,
        'stage1_checkpoint': args.stage1_checkpoint,
        'air_checkpoint': args.air_checkpoint or None,
        'official_root': args.official_root or None,
        'run_evaluator': bool(args.run_evaluator),
        'evaluator_methods': list(evaluator_methods),
        'shard_id': args.shard_id,
        'num_shards': args.num_shards,
        'sensor_depth_role': 'visual context only; never fed to RGB-only model',
    }
    vis.root.mkdir(parents=True, exist_ok=True)
    (vis.root / f'run_shard{args.shard_id}.json').write_text(
        json.dumps(run, indent=2))

    from models.economicgrasp_cva_centers import extract_depth_features
    processed = 0
    for split, sid in scene_jobs:
        ds, lookup = make_dataset(args.dataset_root, split, protocol['camera'])
        for aid in frames:
            frame_base = vis.root / split / f'scene_{sid:04d}' / f'ann_{aid:04d}'
            marker = frame_base / 'completed.json'
            if marker.is_file() and not args.overwrite:
                print(f'[VIS skip] {split} {sid}/{aid}', flush=True)
                continue
            batch = get_batch(ds, lookup, sid, aid, device)
            pack = extract_depth_features(model.reference, batch)
            rgb_model = denormalize_rgb(batch['img'])
            # Sensor depth is context only, aligned to the 448 model crop.
            idx = lookup[(sid, aid)]
            raw_item = ds[idx]
            sensor = np.asarray(raw_item['sensor_depth_m'], np.float32)
            case_results = {}

            with torch.no_grad():
                for case in cases:
                    case_seed = seed_for(2030, sid, aid, case)
                    result = _score_corrector_inspect(
                        model, batch, pack, case, case_seed,
                        args.query_limit, args.query_chunk)
                    result.update({
                        '_split': split,
                        '_scene_id': sid,
                        '_anno_id': aid,
                        '_model': model,
                    })
                    case_results[case] = result
                    out = vis.frame_dir(split, sid, aid, case)
                    air_result = None
                    if air is not None and vis.wants('air'):
                        air_result = air(
                            batch, bundle=result['bundle'], case=case,
                            case_seed=case_seed,
                            query_chunk=args.query_chunk, depth_pack=pack)
                    _save_case(
                        vis, out, rgb_model, sensor, batch['K'], pack,
                        result, case, args, args.official_root, air_result,
                        frame_evaluator, evaluator_methods)

            if vis.wants('grasp_delta') and 'nominal' in case_results:
                nominal = case_results['nominal']
                compare_dir = frame_base / 'cross_case'
                compare_dir.mkdir(parents=True, exist_ok=True)
                for case, cur in case_results.items():
                    if case == 'nominal':
                        continue
                    # Query-id matched Stage-1 native change caused by corruption.
                    b, a = _match_query_actions(
                        nominal, cur,
                        to_numpy(nominal['outputs']['native']),
                        to_numpy(cur['outputs']['native']))
                    if b is not None:
                        save_center_shift_overlay(
                            compare_dir / f'{case.replace(":", "_")}_stage1_change.png',
                            rgb_model, batch['K'], b, a,
                            topk=max(args.topk, 50),
                            title=f'Same-query Stage-1 center change: nominal -> {case}')
                    b, a = _match_query_actions(
                        nominal, cur,
                        to_numpy(nominal['outputs']['stage1']),
                        to_numpy(cur['outputs']['stage1']))
                    if b is not None:
                        save_center_shift_overlay(
                            compare_dir / f'{case.replace(":", "_")}_dcr_change.png',
                            rgb_model, batch['K'], b, a,
                            topk=max(args.topk, 50),
                            title=f'Same-query DCR output change: nominal -> {case}')

            frame_base.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({
                'split': split, 'scene_id': sid, 'anno_id': aid,
                'cases': cases, 'items': list(vis_cfg.items),
            }, indent=2))
            processed += 1
            print(f'[VIS] {split} scene={sid} ann={aid} done={processed}', flush=True)
            del batch, pack, case_results
    print(f'[VIS] complete shard={args.shard_id} frames={processed}', flush=True)


if __name__ == '__main__':
    main()
