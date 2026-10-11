#!/usr/bin/env python3
"""Pre-register an AP-independent Novel diagnostic subset and attach archival AP.

Strict default selection: all 30 Novel scenes × frame IDs (0, 80, 160, 240),
120 unique frames. Frame IDs are selected from the canonical 10% GraspNet
sampling (0,10,...,250), without examining grasp quality or geometry.

Output schema is compatible with export_gvar_probe_sidecars.py and the P1
probe analysis scripts. AP is read only *after* the frame set is frozen.
Inference, training and GraspNet evaluation are never invoked by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_gvar_scene_paired import AuditError, FRAMES, SPLITS, load_all
from diagnose_gvar_novel_failures import frame_ap

DEFAULT_FRAMES = (0, 80, 160, 240)
DEFAULT_SCENES = tuple(range(160, 190))
VERSION = 1


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise AuditError(msg)


def selected_ids(scenes: tuple[int, ...], frames: tuple[int, ...]) -> list[tuple[int, int]]:
    """Must run before any AP tensor is read: selection is geometry/AP blind."""
    require(bool(scenes) and len(set(scenes)) == len(scenes), 'Need unique scenes')
    require(bool(frames) and len(set(frames)) == len(frames), 'Need unique frame IDs')
    require(all(s in DEFAULT_SCENES for s in scenes), 'Novel scene IDs must be 160..189')
    require(all(f in FRAMES for f in frames), 'Frame IDs must be in 0,10,...,250')
    return [(scene, frame) for scene in sorted(scenes) for frame in sorted(frames)]


def _write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), 'Refuse empty CSV')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path, output: Path, *, scenes: tuple[int, ...] = DEFAULT_SCENES,
        frame_ids: tuple[int, ...] = DEFAULT_FRAMES, reference: str = 'baseline',
        comparison: str = 'volume_rel', epoch: int = 19) -> dict:
    require(reference != comparison, 'Reference and comparison must differ')
    require(epoch == 19, 'Only archived e19 Original GraspNet AP supported')
    frozen = selected_ids(scenes, frame_ids)
    require(not output.exists() or not any(output.iterdir()),
            f'Refusing to overwrite nonempty output: {output}')
    require(output.resolve() != root.resolve(), 'Output cannot be experiment root')
    for reserved in ('eval', 'train'):
        require(not output.resolve().is_relative_to((root/reserved).resolve()),
                f'Output cannot be inside {reserved} directory')
    # Read AP *after* selecting scene/frame IDs, then audit all 3 splits,
    # manifest fingerprints, checkpoint identity and original train subset.
    records, audits = load_all(root, [reference, comparison], epoch)
    b = records[reference]['data']['test_novel'].tensor
    c = records[comparison]['data']['test_novel'].tensor
    frame_rows: list[dict] = []
    chosen: list[dict] = []
    for sid, fid in frozen:
        ref_ap, ref04, ref08 = frame_ap(b, sid)
        cmp_ap, cmp04, cmp08 = frame_ap(c, sid)
        pos = FRAMES.index(fid)
        row = {
            'scene_id': sid, 'frame_id': fid, 'reference': reference,
            'comparison': comparison,
            'reference_AP': float(ref_ap[pos]), 'comparison_AP': float(cmp_ap[pos]),
            'delta_AP_pp': float(cmp_ap[pos] - ref_ap[pos]),
            'reference_mu04': float(ref04[pos]), 'comparison_mu04': float(cmp04[pos]),
            'delta_mu04_pp': float(cmp04[pos] - ref04[pos]),
            'reference_mu08': float(ref08[pos]), 'comparison_mu08': float(cmp08[pos]),
            'delta_mu08_pp': float(cmp08[pos] - ref08[pos]),
            'selection': 'uniform',
        }
        frame_rows.append(row)
        chosen.append({k: row[k] for k in (
            'scene_id', 'frame_id', 'selection', 'reference', 'comparison',
            'delta_AP_pp', 'delta_mu04_pp', 'delta_mu08_pp')})
    expected = len(scenes) * len(frame_ids)
    require(len(chosen) == expected, f'Expected {expected} frames; found {len(chosen)}')
    index_blob = '\n'.join(f'scene_{s:04d}/{f:04d}' for s,f in frozen).encode()
    manifest = {
        'version': VERSION, 'root': str(root.resolve()), 'epoch': epoch,
        'split': 'test_novel', 'scene_ids': sorted(scenes),
        'frame_ids': sorted(frame_ids), 'selection_policy': 'pre_registered_fixed_frame_ids',
        'AP_blind_selection': True, 'selection': 'uniform',
        'selection_sha256': hashlib.sha256(index_blob).hexdigest(),
        'variants': [reference,comparison], 'reference': reference,
        'comparison': comparison, 'selected_frames': chosen,
        'selection_count': len(chosen),
        'checkpoint_sha256': {name: records[name]['ckpt_sha256'] for name in (reference,comparison)},
        'train_eval_inputs': [{'path': item['path'], 'sha256': item['sha256']} for item in audits],
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output/'novel_frames.csv', frame_rows)
    (output/'selected_frames.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    report = [
        '# GVAR AP-blind Novel scene-frame diagnostic cohort', '',
        f'Original GraspNet e19; {len(scenes)} scenes × {len(frame_ids)} frames = {len(chosen)} frames.',
        f'Scene IDs: {sorted(scenes)}.', f'Frame IDs: {sorted(frame_ids)}.',
        'Selection uses fixed scene/frame IDs ONLY; grasp AP was read after ID selection.',
        'Cohort is systematic, not random IID frame sampling; adjacent frames and scene clusters are correlated.',
        f'Frame-AP deltas are {comparison} − {reference} for independently generated grasps.',
        'This does not assert fixed-physical-candidate ranking differences or causal effects.', '',
        'Artifacts: `selected_frames.json`, `novel_frames.csv`, plus original SHA256 provenance.',
    ]
    (output/'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    return {'frames': len(chosen), 'scenes': len(scenes), 'frame_ids': sorted(frame_ids),
            'output_dir': str(output.resolve()), 'selection_sha256': manifest['selection_sha256']}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scenes', type=int, nargs='+', default=DEFAULT_SCENES)
    parser.add_argument('--frame-ids', type=int, nargs='+', default=DEFAULT_FRAMES)
    parser.add_argument('--reference', default='baseline')
    parser.add_argument('--comparison', default='volume_rel')
    parser.add_argument('--epoch', type=int, default=19)
    a = parser.parse_args(argv)
    try:
        result = run(a.root, a.output_dir, scenes=tuple(a.scenes),frame_ids=tuple(a.frame_ids),
                     reference=a.reference,comparison=a.comparison,epoch=a.epoch)
    except (AuditError,FileNotFoundError,KeyError,ValueError,OSError) as exc:
        parser.exit(2, f'Uniform Novel selection FAILED: {exc}\n')
    print(json.dumps(result, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
