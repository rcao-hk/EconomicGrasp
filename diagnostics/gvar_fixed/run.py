"""G0/G1/G2: capture native models, freeze physical actions, replay scores."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch

from .contracts import (VERSION, read_frames, file_sha256, write_json, write_csv,
                        exact_labels, make_rows)


def load_factory(name):
    module, symbol = name.split(':', 1)
    return getattr(importlib.import_module(module), symbol)


def main_dataset(root, camera, split, num_points):
    from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
    dataset = GraspNetMultiDataset(root, split='test_' + split, camera=camera,
                                  num_points=num_points, remove_outlier=True,
                                  augment=False, load_label=False)
    return dataset, collate_fn


def dataset_indices(dataset):
    # This is the production inference_cva.py scene-list + 256-frame contract,
    # but checked rather than assumed for arbitrary datasets.
    scenes = dataset.scene_list()
    if len(scenes) != len(dataset) or len(scenes) % 256:
        raise ValueError('Dataset must expose complete 256-frame GraspNet scenes')
    result = {}
    for start in range(0, len(scenes), 256):
        names = scenes[start:start + 256]
        if len(set(names)) != 1:
            raise ValueError('Dataset is not in canonical contiguous scene/frame order')
        scene = int(str(names[0])[-4:])
        for anno in range(256):
            key = (scene, anno)
            if key in result:
                raise ValueError('Duplicate scene/anno mapping')
            result[key] = start + anno
    return result


def git_identity():
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
        return dict(head=sha, dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        return dict(head=None, dirty='not a git working tree')


def run(args):
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a new empty output directory; historical evidence is never overwritten')
    if args.families < 1:
        raise ValueError('families must be positive')
    frames = read_frames(args.frames)
    root.mkdir(parents=True, exist_ok=True)
    configs = {'gvar': json.loads(Path(args.gvar_config).read_text())}
    if args.point_config:
        configs['point'] = json.loads(Path(args.point_config).read_text())
    references = args.references.split(',')
    if len(set(references)) != len(references) or any(x not in configs for x in references):
        raise ValueError('Each reference must have a matching runtime config')
    conditions = ['native', 'rgb_fixed', 'perm_cycle']
    if args.geometry_bias_mm:
        conditions.extend('geometry_bias_' + x.strip() for x in args.geometry_bias_mm.split(','))
    if len(set(conditions)) != len(conditions):
        raise ValueError('Repeated intervention')
    provenance = dict(version=VERSION, status='running', frames=frames,
                      frame_manifest_sha256=file_sha256(args.frames), git=git_identity(),
                      config_sha256={k: file_sha256(args.gvar_config if k == 'gvar' else args.point_config) for k in configs},
                      runtime_configs=configs, seed=args.seed, families=args.families,
                      references=references, conditions=conditions,
                      camera=args.camera, dataset_root=str(Path(args.dataset_root).resolve()),
                      dataset_factory=args.dataset_factory, fc_mode='official', collision_prefilter=False,
                      network_depth_source='RGB_prediction',
                      preprocessing='native dataset workspace/crop may use sensor depth; not raw-RGB-only pipeline',
                      files=[])
    write_json(root / 'run_manifest.json', provenance)
    try:
        runtimes = {}
        for model, conf in configs.items():
            factory = conf.get('factory', 'diagnostics.gvar_fixed.runtime:build_main_runtime')
            runtimes[model] = load_factory(factory)(conf, args.device)
            if runtimes[model].checkpoint_sha != conf.get('expected_sha256'):
                raise ValueError('Historical runtime must return its verified actual checkpoint SHA256')
            if (runtimes[model].reader is not None) != (model == 'gvar'):
                raise ValueError(f'{model} config does not match actual reader presence')
        counts = {r.settings['num_points'] for r in runtimes.values()}
        if len(counts) != 1:
            raise ValueError('Point sampling/preprocessing settings must match across models')
        factory = load_factory(args.dataset_factory)
        datasets, evaluators = {}, {}
        from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
        for split in sorted({f['split'] for f in frames}):
            with runtimes['gvar'].activation():
                ds, collate = factory(args.dataset_root, args.camera, split, next(iter(counts)))
            datasets[split] = ds, collate, dataset_indices(ds)
            evaluators[split] = ExactGraspNetActionEvaluator(args.dataset_root, args.camera,
                                                            split='test_' + split,
                                                            fc_mode='official', strict=True,
                                                            collision_chunk=args.eval_chunk)
        all_rows = []
        for frame in frames:
            local_seed = args.seed + frame['scene'] * 1009 + frame['anno'] * 101
            random.seed(local_seed); np.random.seed(local_seed); torch.manual_seed(local_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(local_seed)
            ds, collate, index = datasets[frame['split']]
            key = frame['scene'], frame['anno']
            if key not in index:
                raise ValueError(f'Requested frame not present: {key}')
            with runtimes['gvar'].activation():
                batch = collate([ds[index[key]]])
            captured = {name: runtime.capture(batch) for name, runtime in runtimes.items()}
            audits = {'native_parity': {name: r.validate_native(captured[name], args.parity_atol)
                                        for name, r in runtimes.items()}, 'replay': []}
            frame_rows = []
            for reference in references:
                snapshot = captured[reference]
                bank, query_ids, angle_ids = runtimes[reference].make_bank(snapshot, args.families, local_seed)
                labels = exact_labels(evaluators[frame['split']], frame['scene'], frame['anno'], bank)
                bank_path = root / 'banks' / f'{frame["sample"]}_{reference}.npz'
                bank_path.parent.mkdir(exist_ok=True)
                np.savez_compressed(bank_path, grasps=bank, query_ids=query_ids,
                                    angle_ids=angle_ids, quality=labels['quality'],
                                    success=labels['success'], empty=labels['empty'],
                                    collision=labels['collision'], friction=labels['friction'])
                for model, runtime in runtimes.items():
                    for condition in conditions if model == 'gvar' else ['native']:
                        predictions, audit = runtime.replay(captured[model], snapshot, condition)
                        scores, widths = runtime.selected_outputs(predictions, query_ids, angle_ids)
                        rows = make_rows(frame, reference, bank, query_ids, angle_ids,
                                         labels, scores, model, condition, snapshot.geometry_hash,
                                         runtime.checkpoint_sha, snapshot.checkpoint_sha, widths)
                        frame_rows.extend(rows)
                        audits['replay'].append(dict(reference=reference, model=model, **audit))
                audits.setdefault('banks', {})[reference] = dict(
                    sha256=file_sha256(bank_path), geometry_sha256=snapshot.geometry_hash,
                    evaluator_stats=labels['stats'])
            path = root / 'frames' / f'{frame["sample"]}.csv'
            write_csv(path, frame_rows)
            write_json(path.with_suffix('.audit.json'), audits)
            provenance['files'].append(dict(path=str(path.relative_to(root)), sha256=file_sha256(path)))
            write_json(root / 'run_manifest.json', provenance)
            all_rows.extend(frame_rows)
            print(f'[GVAR fixed] {frame["sample"]}: {len(frame_rows)} score records; native parity passed', flush=True)
            del captured, batch
        write_csv(root / 'scores.csv', all_rows)
        provenance.update(status='complete', score_csv_sha256=file_sha256(root / 'scores.csv'),
                          rows=len(all_rows), completed_frames=len(frames))
        write_json(root / 'run_manifest.json', provenance)
    except Exception as exc:
        provenance.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(root / 'run_manifest.json', provenance)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gvar-config', required=True)
    p.add_argument('--point-config', default=None)
    p.add_argument('--frames', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--camera', default='realsense')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--references', default='point,gvar')
    p.add_argument('--families', type=int, default=16)
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--parity-atol', type=float, default=1e-5)
    p.add_argument('--eval-chunk', type=int, default=256)
    p.add_argument('--geometry-bias-mm', default='', help='Optional GVAR-only relative-depth stress, e.g. -5,5')
    p.add_argument('--dataset-factory', default='diagnostics.gvar_fixed.run:main_dataset')
    run(p.parse_args())


if __name__ == '__main__':
    main()
