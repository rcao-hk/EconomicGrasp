"""Physical action banks and exact-label contracts (no model imports)."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

MUS = np.array([.2, .4, .6, .8, 1., 1.2], dtype=np.float64)
VERSION = 'gvar-fixed-actions-v1'


def digest_array(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.shape).encode()); h.update(a.dtype.str.encode()); h.update(a.tobytes())
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def physical_id(row: np.ndarray) -> str:
    a = np.array(row, dtype='<f4', copy=True)
    if a.shape != (17,) or not np.isfinite(a).all():
        raise ValueError('A physical grasp must be a finite 17D GraspNet row')
    a[a == 0] = 0  # normalize negative zero, without rounding physical coordinates
    return hashlib.sha256(a[1:16].tobytes()).hexdigest()[:24]


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(safe_json(value), ensure_ascii=False, indent=2,
                              allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('Refusing to write an empty result table')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(set(x) != set(fields) for x in rows):
        raise ValueError('CSV rows have inconsistent schema')
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def read_frames(path: str | Path) -> list[dict]:
    with open(path, newline='', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        if not {'scene_id', 'anno_id', 'split'} <= set(rd.fieldnames or []):
            raise ValueError('Frame manifest requires scene_id,anno_id,split')
        rows = list(rd)
    if not rows:
        raise ValueError('Empty frame manifest')
    result, seen = [], set()
    for r in rows:
        scene, anno = int(r['scene_id']), int(r['anno_id'])
        split = r['split'].removeprefix('test_')
        inferred = 'seen' if 100 <= scene < 130 else 'similar' if 130 <= scene < 160 else 'novel' if 160 <= scene < 190 else None
        if split != inferred or not 0 <= anno < 256:
            raise ValueError(f'Unsupported GraspNet test frame: {r}')
        if (scene, anno) in seen:
            raise ValueError('Duplicate scene/frame in manifest')
        seen.add((scene, anno))
        result.append(dict(scene=scene, anno=anno, split=split,
                           sample=f'scene{scene:04d}_frame{anno:04d}'))
    return result


def validate_grasps(grasps: np.ndarray, max_width: float = .1) -> None:
    a = np.asarray(grasps)
    if a.ndim != 3 or a.shape[1:] != (4, 17) or not np.isfinite(a).all():
        raise ValueError('Bank must be finite [families,4,17]')
    if np.any(a[..., 1] < 0) or np.any(a[..., 1] > max_width + 1e-6):
        raise ValueError('Physical executed width violates the configured cap')
    if not np.allclose(a[..., 3], np.arange(1, 5) * .01, atol=1e-7, rtol=0):
        raise ValueError('Expected insertion depths 1/2/3/4 cm in fixed order')
    if not np.allclose(a[..., 13:16], a[:, :1, 13:16], atol=1e-7, rtol=0):
        raise ValueError('Center changed within insertion family')
    if not np.allclose(a[..., 4:13], a[:, :1, 4:13], atol=1e-7, rtol=0):
        raise ValueError('Orientation changed within insertion family')
    R = a[..., 4:13].reshape(-1, 3, 3)
    if not np.allclose(R.transpose(0, 2, 1) @ R, np.eye(3), atol=1e-4, rtol=0) or np.any(np.linalg.det(R) < .999):
        raise ValueError('Invalid SO(3) matrices in action bank')


def exact_labels(evaluator, scene: int, anno: int, grasps: np.ndarray) -> dict:
    """Evaluate each distinct physical action once; never prefilter collisions."""
    flat = np.asarray(grasps, dtype=np.float32).reshape(-1, 17)
    unique = {}
    for row in flat:
        unique.setdefault(physical_id(row), row)
    ids = sorted(unique)
    result = evaluator.evaluate(scene, anno, np.stack([unique[k] for k in ids]))
    idx = {k: i for i, k in enumerate(ids)}
    order = np.array([idx[physical_id(r)] for r in flat])
    friction = np.asarray(result.friction, dtype=float)[order]
    empty = np.asarray(result.empty, dtype=bool)[order]
    collision = np.asarray(result.pure_collision, dtype=bool)[order]
    combined = np.asarray(result.collision_or_empty, dtype=bool)[order]
    if not np.isfinite(friction).all() or np.any(combined != (empty | collision)):
        raise ValueError('Malformed exact evaluator labels')
    if np.any((empty | collision) & (friction > 0)):
        raise ValueError('Collision/empty action cannot have positive friction quality')
    bits = (friction[:, None] > 0) & (friction[:, None] <= MUS[None] + 1e-7)
    return dict(friction=friction.reshape(grasps.shape[:2]),
                empty=empty.reshape(grasps.shape[:2]),
                collision=collision.reshape(grasps.shape[:2]),
                success=bits.reshape(*grasps.shape[:2], 6),
                quality=bits.mean(-1).reshape(grasps.shape[:2]),
                stats=getattr(result, 'stats', {}))


def bank_id(grasps: np.ndarray, geometry_hash: str, reference: str) -> str:
    # Score column is deliberately excluded from physical identity.
    return hashlib.sha256((digest_array(grasps[..., 1:16]) + geometry_hash + reference).encode()).hexdigest()


def make_rows(frame, reference, bank, query_ids, angle_ids, labels, scores,
              model, condition, geometry_hash, checkpoint_sha, source_sha,
              predicted_widths=None):
    validate_grasps(bank)
    if np.shape(scores) != bank.shape[:2] or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite and cover every frozen action')
    uid = bank_id(bank, geometry_hash, reference)
    rows = []
    for f, q in enumerate(query_ids):
        for d in range(4):
            g = bank[f, d]
            failure = ('empty_grasp' if labels['empty'][f, d] else
                       'collision' if labels['collision'][f, d] else
                       'success' if labels['quality'][f, d] > 0 else 'force_closure_failure')
            rows.append(dict(
                version=VERSION, sample_id=frame['sample'], scene_id=frame['scene'],
                anno_id=frame['anno'], split=frame['split'], reference=reference,
                bank_id=uid, geometry_sha256=geometry_hash,
                model=model, condition=condition, checkpoint_sha256=checkpoint_sha,
                source_checkpoint_sha256=source_sha,
                family_id=f'q{int(q):04d}_a{int(angle_ids[f]):02d}',
                query_id=int(q), angle_id=int(angle_ids[f]), depth_id=d,
                candidate_id=physical_id(g), score=float(scores[f, d]),
                quality=float(labels['quality'][f, d]), failure=failure,
                friction=float(labels['friction'][f, d]),
                width_m=float(g[1]), height_m=float(g[2]), insertion_depth_m=float(g[3]),
                center_x=float(g[13]), center_y=float(g[14]), center_z=float(g[15]),
                **{f'r{i}{j}': float(g[4 + i * 3 + j]) for i in range(3) for j in range(3)},
                **{f'success_mu_{mu:.1f}': int(labels['success'][f, d, t]) for t, mu in enumerate(MUS)},
                width_head_output_m=float(predicted_widths[f, d]) if predicted_widths is not None else '',
                executed_width_source='frozen_reference_native_per_depth',
                score_is_explicitly_width_conditioned=0,
            ))
    return rows
