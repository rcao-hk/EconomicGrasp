"""Cross-pool audit + controlled 2x2 training. No backbone/CAD training.

Primary artifact is always fixed epoch10, never test-AP-selected. Validation
and confirmation use the SAME common pools for all four arms. Original 90–99
scenes have already been inspected: they are a reused audit, not fresh evidence.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from .core import (ARMS, BASE_SHA, SCHEMA, QualityProbe, digest, loss, metrics,
                   predict, sha, subset, write_json)
from .data import source_index, union_frame, load_plan, read_pack, frame_name


def load_head(spec, device):
    if not spec or not spec.get('sha256') or sha(spec['file']) != spec['sha256']:
        raise ValueError('Trusted checkpoint file and matching SHA256 required')
    ck = torch.load(spec['file'], map_location='cpu', weights_only=False)
    if ck.get('schema') not in (SCHEMA, 'quality-decision-p0p2-v1'):
        raise ValueError('Unknown scorer schema; do not auto-map an arbitrary checkpoint')
    c = ck['config']
    if c['base_checkpoint_sha256'] != BASE_SHA or c.get('test_labels_used') is not False:
        raise ValueError('Wrong base or test-label use not explicitly excluded')
    if c['hidden'] != 64 or c['mode'] != 'score_feature_geometry':
        raise ValueError('This factorial audit requires the matched 5,633-parameter head')
    model = QualityProbe(c['mode'], c['hidden']).to(device)
    model.load_state_dict(ck['model'], strict=True); model.eval().requires_grad_(False)
    if sum(p.numel() for p in model.parameters()) != 5633: raise RuntimeError('Capacity mismatch')
    return model, ck


def aggregate(rows):
    """Equal scenes, then domains, then frames. Missing comparisons stay missing."""
    keys = [k for k in rows[0] if k not in ('scene', 'anno', 'domain', 'pool', 'model')]
    result = {}
    for k in keys:
        scene_values = []
        for scene in sorted({r['scene'] for r in rows}):
            domains = []
            for d in ('original', 'gntrans'):
                v = [r[k] for r in rows if r['scene'] == scene and r['domain'] == d and r[k] is not None]
                if v: domains.append(float(np.mean(v)))
            if domains: scene_values.append(float(np.mean(domains)))
        result[k] = float(np.mean(scene_values)) if scene_values else None
    return result


def write_csv(path, rows):
    if not rows: raise ValueError('Empty output')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


@torch.inference_mode()
def evaluate(models, frames, device):
    result = []
    for row, x in frames:
        for pool in ('uniform', 'decision', 'union'):
            fam = np.unique(x['family']) if pool == 'union' else x[pool + '_families']
            bank = subset(x, fam)
            for name, model in models.items():
                score = (torch.as_tensor(bank['base_logit']).sigmoid().numpy() if model is None else
                         predict(model, bank, device).sigmoid().cpu().numpy())
                result.append(dict(scene=int(row['scene']), anno=int(row['anno']), domain=row['domain'],
                                   pool=pool, model=name, **metrics(score, bank)))
    return result


def cross(args):
    specs = json.loads(Path(args.heads).read_text())
    if len(specs) != 4: raise ValueError('Require exactly four named scorer checkpoints')
    models = {'native': None}; ckpts = {}
    for name, spec in specs.items():
        if name == 'native': raise ValueError('Reserved model name')
        models[name], ckpts[name] = load_head(spec, args.device)
    if args.plan:
        p = load_plan(args.plan, require_formal=not args.smoke)
        identity = p['source_index_sha256']; config_digest = p['base_config_digest']
        entries = [e for e in p['entries'] if 90 <= int(e['row']['scene']) < 100]
        frames = [(e['row'], read_pack(e)) for e in entries]
        provenance = {'plan_sha256': p['plan_sha256']}
    else:
        src, paired = source_index(args.source_index, formal=not args.smoke)
        identity = src['index_sha256']; config_digest = src['config_digest']
        frames = [(es['uniform']['row'], union_frame(es)) for key, es in paired if 90 <= key[1] < 100]
        provenance = {'source_index_sha256': identity}
    if not frames or (not args.smoke and len(frames) != 80): raise ValueError('Require 80 reused audit views')
    for name, ck in ckpts.items():
        c = ck['config']
        source = c.get('source_index_sha256', c.get('cache_contract', {}).get('index_sha256'))
        if source != identity or c['base_config_digest'] != config_digest:
            raise ValueError(f'{name}: source cache/runtime differs; do not silently ignore identity')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = evaluate(models, frames, args.device); write_csv(out / 'per_frame.csv', rows)
    summary = {}
    for pool in ('uniform', 'decision', 'union'):
        summary[pool] = {name: aggregate([r for r in rows if r['pool'] == pool and r['model'] == name])
                         for name in models}
    write_json(out / 'summary.json', dict(schema=SCHEMA, kind='common-pool-cross-audit',
               frames=len(frames), checkpoints=specs, **provenance, metrics=summary,
               reused_audit_scenes=True, scope='sampled pre-sensor-filter/pre-NMS pool, not official Top10/AP'))
    lines = ['# 四评分头 × 相同候选池：交叉审计', '',
             '90–99场景已被查看过；以下是样本池内、传感器过滤及NMS前指标，不是新的独立验证或官方AP。', '',
             '| Pool | Head | MSE | Selected q | Fixed-depth Top10 zero |', '|---|---|---:|---:|---:|']
    for pool, values in summary.items():
        for name, v in values.items():
            lines.append(f"| {pool} | {name} | {v['mse']:.6f} | {v['selected_q']:.6f} | {v['fixed_depth_top10_zero']:.6f} |")
    (out / 'REPORT_CN.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def save_checkpoint(path, model, config, epoch, opt):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(schema=SCHEMA, config=config, epoch=epoch, model=model.state_dict(),
                    optimizer=opt.state_dict()), tmp)
    os.replace(tmp, path)


def fit(args):
    p = load_plan(args.plan, require_formal=not args.smoke)
    if not args.smoke and (args.epochs != 10 or args.batch_frames != 4 or args.seed not in p['seeds']):
        raise ValueError('Formal protocol freezes 10 epochs and the three declared seeds')
    if args.epochs < 1 or args.batch_frames < 1: raise ValueError('Invalid training budget')
    sample, pair = ARMS[args.arm]
    train = [e for e in p['entries'] if int(e['row']['scene']) < 80]
    val = [e for e in p['entries'] if 80 <= int(e['row']['scene']) < 90]
    if not args.smoke and (len(train), len(val)) != (640, 80): raise ValueError('640/80 views required')
    if not train or not val: raise ValueError('Empty training/validation')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model = QualityProbe().to(args.device)
    init = digest({k: sha_tensor(v) for k, v in model.state_dict().items()})
    config = dict(mode='score_feature_geometry', hidden=64, arm=args.arm, sampler=sample,
                  pair_mode=pair, seed=args.seed, epochs=args.epochs, batch_frames=args.batch_frames,
                  base_checkpoint_sha256=BASE_SHA, base_config_digest=p['base_config_digest'],
                  source_index_sha256=p['source_index_sha256'], plan_sha256=p['plan_sha256'],
                  train_scenes=[0, 80], val_scenes=[80, 90], test_labels_used=False,
                  primary_selection='fixed-last-epoch-not-MSE-or-test-AP',
                  learning_rate=1e-4, weight_decay=1e-4, point_weight=10., pair_weight=.5,
                  initial_state_sha256=init, train_frames=len(train), validation_frames=len(val),
                  smoke=bool(args.smoke), source_sha256=sha(__file__))
    write_json(out / 'protocol.json', config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed); history = []; steps = 0
    # Validation is identical union pool across all four arms, no label-stratified validation.
    val_frames = [(e['row'], read_pack(e)) for e in val]
    for epoch in range(args.epochs + 1):
        audit_rows = []
        if epoch:
            model.train(); order = rng.permutation(len(train))
            for start in range(0, len(order), args.batch_frames):
                optimizer.zero_grad(set_to_none=True); batch = []
                for i in order[start:start + args.batch_frames]:
                    e = train[int(i)]; x = read_pack(e)
                    families = x['decision_families'] if sample == 'decision' else e['stratified_families']
                    bank = subset(x, families)
                    if len(bank['count']) != 256: raise ValueError('Training budget changed')
                    z = predict(model, bank, args.device)
                    l, audit = loss(z, bank, pair, args.seed + epoch * 1000003 + int(i) * 1009)
                    if not torch.isfinite(l): raise FloatingPointError('Nonfinite loss')
                    batch.append(l)
                    audit_rows.append(dict(epoch=epoch, frame=frame_name(e['row']), **audit))
                value = torch.stack(batch).mean(); value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); steps += 1
        model.eval()
        rows = evaluate({'probe': model}, val_frames, args.device)
        per_pool = {pool: aggregate([r for r in rows if r['pool'] == pool]) for pool in ('uniform', 'decision', 'union')}
        history.append(dict(epoch=epoch, optimizer_steps=steps, validation=per_pool))
        if audit_rows: write_csv(out / f'pairs_epoch{epoch:02d}.csv', audit_rows)
        save_checkpoint(out / 'last.pt', model, config, epoch, optimizer)
        write_json(out / 'history.json', history)
        print(f"[{args.arm}/seed{args.seed}] epoch={epoch} val-common-MSE={per_pool['union']['mse']:.6f}", flush=True)
    write_json(out / 'complete.json', dict(schema=SCHEMA, arm=args.arm, seed=args.seed,
               epochs=args.epochs, steps=steps, weights_sha256=sha(out / 'last.pt'),
               plan_sha256=p['plan_sha256'], smoke=bool(args.smoke)))


def sha_tensor(v):
    import hashlib
    return hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__); sp = p.add_subparsers(dest='cmd', required=True)
    a = sp.add_parser('cross'); a.add_argument('--heads', required=True)
    g = a.add_mutually_exclusive_group(required=True); g.add_argument('--source-index'); g.add_argument('--plan')
    a.add_argument('--output', required=True); a.add_argument('--device', default='cpu'); a.add_argument('--smoke', action='store_true')
    a = sp.add_parser('fit'); a.add_argument('--plan', required=True); a.add_argument('--arm', choices=tuple(ARMS), required=True)
    a.add_argument('--output', required=True); a.add_argument('--device', default='cuda:0')
    a.add_argument('--seed', type=int, default=20261113); a.add_argument('--epochs', type=int, default=10)
    a.add_argument('--batch-frames', type=int, default=4); a.add_argument('--smoke', action='store_true')
    x = p.parse_args(); (cross if x.cmd == 'cross' else fit)(x)

if __name__ == '__main__': main()
