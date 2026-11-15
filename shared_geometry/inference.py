"""Actual action updates; recompute sensor filtering after EVERY geometry change."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .data import (SCHEMA, load_signed, validate_plan, signed, atomic_json, atomic_npz,
                   frame_key, physical_ids, sha256, code_hash)
from .geometry import action_bank
from .runtime import create_runtime
from .experiment import load_model, predict

POLICIES = ('native', 'field_score_native', 'move_keep_priority', 'move_rescore', 'stale_move_keep_priority')


def instantiate(bank, scores, stale_scores, policy):
    n = len(bank); i = np.arange(n)
    if policy == 'native': return bank[:, 0].copy(), np.zeros(n, dtype=np.int64)
    if policy == 'field_score_native':
        out = bank[:, 0].copy(); out[:, 0] = scores[:, 0]
        return out, np.zeros(n, dtype=np.int64)
    choice = (stale_scores if policy == 'stale_move_keep_priority' else scores).argmax(1)
    out = bank[i, choice].copy()
    out[:, 0] = scores[i, choice] if policy == 'move_rescore' else bank[:, 0, 0]
    return out, choice


def infer(a):
    plan = load_signed(a.plan); validate_plan(plan)
    model, p = load_model(a.weight, a.device, bool(a.limit))
    runtime = create_runtime(a)
    if p['source'] != runtime.provenance or p['plan'] != plan['fingerprint']:
        raise ValueError('Training/inference frozen source or frame plan mismatch')
    if not a.limit and not a.reference_root:
        raise ValueError('Formal inference requires original frozen-MSE reference dumps')
    if a.torch_threads > 0: torch.set_num_threads(a.torch_threads)
    root = Path(a.output)
    if root.exists() and any(root.iterdir()): raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = [r for r in plan['test'] if r['domain'] == 'original' and r['split'] == a.split]
    if a.limit: rows = rows[:a.limit]
    records = []; maxerr = 0.
    # No TargetLoader and no evaluator labels are instantiated in this function.
    for step, row in enumerate(rows):
        arrays, parity = runtime.snapshot(row, a.seed)
        oldkeep = runtime.sensor_keep(row, arrays['grasps']).astype(bool)
        if a.reference_root:
            path = Path(a.reference_root)/f"scene_{row['scene']:04d}"/'realsense'/f"{row['anno']:04d}.npy"
            old = np.load(path, allow_pickle=False); now = arrays['grasps'][oldkeep]
            if old.shape != now.shape: raise RuntimeError('Frozen reference count mismatch')
            err = float(np.max(np.abs(old-now))) if len(now) else 0.
            if not np.isfinite(err) or err > 1e-6: raise RuntimeError('Frozen source parity failure')
            maxerr = max(maxerr, err)
        bank = action_bank(arrays['grasps'], p['offsets_m'])
        frame = dict(arrays, bank=bank, local_features=arrays['features'])
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_conv = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            with torch.inference_mode():
                logits, _ = predict(model, p['mode'], frame, a.device)
                scores = logits.sigmoid().cpu().numpy()
                if p['mode'] in ('h1', 'h2'):
                    stale_logits, _ = predict(model, p['mode'], frame, a.device, 'stale')
                    stale = stale_logits.sigmoid().cpu().numpy()
                else: stale = scores.copy()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32; torch.backends.cudnn.allow_tf32 = old_conv
        if not np.isfinite(scores).all() or not np.isfinite(stale).all(): raise FloatingPointError('Nonfinite prediction')
        trace = dict(native=arrays['grasps'], native_ids=physical_ids(arrays['grasps']), scores=scores, stale_scores=stale)
        for policy in POLICIES:
            out, choice = instantiate(bank, scores, stale, policy)
            if not np.array_equal(out[:, 1:13], arrays['grasps'][:, 1:13]):
                raise RuntimeError('Translation-only refinement changed rotation/width/insertion')
            keep = oldkeep if policy in ('native', 'field_score_native') else runtime.sensor_keep(row, out).astype(bool)
            dest = root/policy/f"scene_{row['scene']:04d}"/'realsense'/f"{row['anno']:04d}.npy"
            dest.parent.mkdir(parents=True, exist_ok=True); np.save(dest, out[keep])
            records.append(dict(row=row, variant=policy, path=str(dest.relative_to(root)), sha256=sha256(dest),
                                queries=len(out), survivors=int(keep.sum()), moved=int((choice != 0).sum())))
            trace[policy+'_choice'] = choice; trace[policy+'_sensor_keep'] = keep
            trace[policy+'_physical_id'] = physical_ids(out)
        atomic_npz(root/'traces'/(frame_key(row)+'.npz'), trace)
        print(f'[infer] {step+1}/{len(rows)} {frame_key(row)}', flush=True)
    atomic_json(root/'inference_manifest.json', signed(dict(schema=SCHEMA, status='complete',
        split=a.split, domain='original', frames=len(rows), partial_smoke=bool(a.limit), variants=list(POLICIES),
        source=runtime.provenance, source_reference_max_abs=maxerr, source_code_sha256=code_hash(),
        weights=dict(path=str(Path(a.weight).resolve()), sha256=sha256(a.weight)), protocol=p, records=records,
        note='Raw learned-policy diagnostics, not a deployment safety certificate; masks recomputed on changed actions')))


def evaluate(a):
    # Reuse the unchanged official evaluation wrapper only after checking our schema.
    m = load_signed(Path(a.inference)/'inference_manifest.json')
    if m['schema'] != SCHEMA: raise ValueError('Wrong inference schema')
    from interactive_grasp.infer import evaluate as official_evaluate
    official_evaluate(a)
