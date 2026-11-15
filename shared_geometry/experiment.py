"""Training and verifiable family evaluation; no test-scene model selection."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
import torch
from .data import (SCHEMA, read_index, read_frame, tensor_scene, collate_frames, code_hash,
                   atomic_json, signed, sha256, load_signed, digest)
from .model import ModelConfig, FieldGraspModel, LocalControl, PredictedGeometryControl, quality_loss, geometry_loss

MODES = ('local', 'predicted', 'h1', 'h2', 'h2_stale')


def predict(model, mode, a, device, intervention='aligned'):
    """Shared B=1/B>1 head path. A batch is NOT a new scene memory.

    Only explicit model-input fields may enter the model. Train labels remain
    separated from the scene dictionary and cannot be read by the field.
    """
    bank = torch.as_tensor(a['bank'], dtype=torch.float32, device=device)
    if mode == 'local':
        features = torch.as_tensor(a['local_features'], dtype=torch.float32, device=device)
        return model(features, bank), None
    scene = tensor_scene(a, device)
    if mode == 'predicted':
        return model(scene, bank), None
    if intervention == 'appearance_mean':
        scene['feature_map'] = scene['feature_map'].mean((-2, -1), keepdim=True).expand_as(scene['feature_map'])
    elif intervention == 'appearance_shuffle':
        fmap = scene['feature_map']; n = fmap.shape[-2]*fmap.shape[-1]
        gen = torch.Generator(device='cpu').manual_seed(119)
        ids = torch.randperm(n, generator=gen).to(device)
        # Same fixed within-image perturbation at B=1 and B>1.
        scene['feature_map'] = (fmap.flatten(1)[:, ids].reshape_as(fmap) if fmap.ndim == 3 else
                                fmap.flatten(2)[:, :, ids].reshape_as(fmap))
    elif intervention not in ('aligned', 'stale'):
        raise ValueError('Unknown intervention')
    enc = model.field.encode(scene)
    access = 'stale' if mode == 'h2_stale' or intervention == 'stale' else 'fresh'
    return model.action_scores(enc, scene, bank, access), (enc, scene)


def family_metrics(z, counts):
    s = np.asarray(z); q = np.asarray(counts, dtype=float)/6
    # Index 0 native wins exact score ties; no intervention forces movement.
    choice = s.argmax(-1); rows = np.arange(len(s)); selected = q[rows, choice]
    native = q[:, 0]; oracle = q.max(-1)
    pair_acc = []; pair_total = 0
    for x, y in zip(s, q):
        i, j = np.triu_indices(len(x), 1); valid = y[i] != y[j]
        i, j = i[valid], j[valid]; pair_total += len(i)
        if len(i):
            value = np.sign(x[i]-x[j])*np.sign(y[i]-y[j])
            pair_acc.append(float(np.mean(np.where(value == 0, .5, value > 0))))
    return dict(native_q=float(native.mean()), selected_q=float(selected.mean()), oracle_q=float(oracle.mean()),
                gain=float((selected-native).mean()), regret=float((oracle-selected).mean()),
                rescue=float(np.mean(selected > native)), harm=float(np.mean(selected < native)),
                move=float(np.mean(choice != 0)), comparable_pairs=pair_total,
                pair_accuracy=float(np.mean(pair_acc)) if pair_acc else None,
                quality_mse=float(np.mean((1/(1+np.exp(-np.clip(s, -60, 60)))-q)**2)))


def balanced(rows, key):
    values = []
    for domain in sorted({r['row']['domain'] for r in rows}):
        scenes = []
        for scene in sorted({r['row']['scene'] for r in rows if r['row']['domain'] == domain}):
            v = [r['metrics'][key] for r in rows if r['row']['domain'] == domain and r['row']['scene'] == scene
                 and r['metrics'].get(key) is not None]
            if v: scenes.append(float(np.mean(v)))
        if scenes: values.append(float(np.mean(scenes)))
    return float(np.mean(values)) if values else None


@torch.no_grad()
def evaluate_index(model, mode, idx, device, intervention='aligned', batch_size=1):
    """True-batch head evaluation; legacy metric aggregation remains per image."""
    if batch_size < 1:
        raise ValueError('Validation batch size must be positive')
    model.eval(); rows = []
    entries = idx['entries']
    from .geometry import VisibleSurfaceTeacher
    for start in range(0, len(entries), batch_size):
        portion = entries[start:start+batch_size]
        frames = [read_frame(entry) for entry in portion]
        inputs = frames[0] if len(frames) == 1 else collate_frames(frames)
        z, field = predict(model, mode, inputs, device, intervention)
        predictions = z.cpu().numpy()
        if len(frames) == 1:
            predictions = predictions[None]
        if predictions.shape[:1] != (len(frames),):
            raise RuntimeError('Head returned the wrong number of scenes')
        physical = None
        if field is not None:
            encoded, scene = field
            xyz = torch.as_tensor(np.stack([f['target_xyz'] for f in frames]),
                                  device=device, dtype=torch.float32)
            # For B=1 preserve original caller shape (P,3).
            if len(frames) == 1:
                xyz = xyz[0]
            raw = model.field.query(encoded, scene, xyz)
            if len(frames) == 1:
                raw = raw[None]
            physical = raw.detach().cpu().numpy()
        for i, (entry, a) in enumerate(zip(portion, frames)):
            metrics = family_metrics(predictions[i], a['quality_count'])
            prior = VisibleSurfaceTeacher(a['pred_depth'], a['K'], idx['contract']['tau']).label(a['target_xyz'])
            um, pm = a['target_udf_mask'], a['target_projective_mask']
            tau = idx['contract']['tau'] * 1000
            metrics.update(predicted_udf_mae_mm=float(np.abs(prior['udf'][um]-a['target_udf'][um]).mean()*tau) if um.any() else None,
                           predicted_projective_mae_mm=float(np.abs(prior['projective'][pm]-a['target_projective'][pm]).mean()*tau) if pm.any() else None,
                           predicted_state_accuracy=float(np.mean(prior['state']==a['target_state'])))
            if physical is not None:
                r = physical[i]; ud = 1/(1+np.exp(-np.clip(r[:,0],-60,60)))
                pr = np.tanh(r[:,1]); near = um & (a['target_udf'] < .25)
                tau = model.cfg.truncation_m * 1000
                metrics.update(udf_mae_mm=float(np.abs(ud[um]-a['target_udf'][um]).mean()*tau) if um.any() else None,
                    near_surface_udf_mae_mm=float(np.abs(ud[near]-a['target_udf'][near]).mean()*tau) if near.any() else None,
                    projective_mae_mm=float(np.abs(pr[pm]-a['target_projective'][pm]).mean()*tau) if pm.any() else None,
                    state_accuracy=float((r[:,2:].argmax(-1) == a['target_state']).mean()))
            rows.append(dict(row=entry['row'], metrics=metrics))
    keys = sorted({k for r in rows for k in r['metrics']})
    return dict(summary={k: balanced(rows, k) for k in keys}, rows=rows)


def fit(a):
    batch_size = int(getattr(a, 'batch_size', 1))
    accumulation = int(a.accumulate)
    val_batch_size = int(getattr(a, 'val_batch_size', batch_size))
    if (a.mode not in MODES or a.epochs < 1 or batch_size < 1 or accumulation < 1 or
            val_batch_size < 1 or a.lr <= 0 or a.geometry_weight < 0):
        raise ValueError('Invalid mode/training budget or batch sizes')
    tr = read_index(a.train_index, 'train', a.smoke)
    va = read_index(a.val_index, 'validation', a.smoke)
    for key in ('source', 'plan', 'offsets_m', 'tau', 'band', 'target_config'):
        if tr['contract'][key] != va['contract'][key]:
            raise ValueError('Training/validation contract differs: '+key)
    if {e['row']['scene'] for e in tr['entries']} & {e['row']['scene'] for e in va['entries']}:
        raise ValueError('Train/validation scene leakage')
    if not a.smoke and (len(tr['entries']) != 2560 or len(va['entries']) != 160):
        raise ValueError('Formal phase uses 2560 training and 160 validation frames')
    root = Path(a.output)
    if root.exists() and any(root.iterdir()) and not a.resume:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.cpu_threads); torch.manual_seed(a.seed)
    random.seed(a.seed); np.random.seed(a.seed)
    # New heads FP32; frozen provider precision is independently recorded.
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    first = read_frame(tr['entries'][0])
    cfg = ModelConfig(first['feature_map'].shape[0], a.hidden, tr['contract']['tau'])
    model = (LocalControl(first['local_features'].shape[1], a.hidden) if a.mode == 'local' else
             PredictedGeometryControl(cfg) if a.mode == 'predicted' else FieldGraspModel(cfg)).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    effective_batch = batch_size * accumulation
    settings = dict(mode=a.mode, epochs=a.epochs, batch_size=batch_size,
                    accumulate=accumulation, effective_batch_size=effective_batch,
                    val_batch_size=val_batch_size, seed=a.seed,
                    lr=a.lr, weight_decay=a.weight_decay, geometry_weight=a.geometry_weight,
                    smoke=bool(a.smoke), local_feature_dim=first['local_features'].shape[1], config=asdict(cfg),
                    train_fingerprint=tr['fingerprint'], val_fingerprint=va['fingerprint'],
                    source=tr['contract']['source'], offsets_m=tr['contract']['offsets_m'],
                    plan=tr['contract']['plan'], source_code_sha256=code_hash(),
                    selection='maximum scene-balanced validation selected-Q gain; ties earliest; retain native if <=0',
                    trainable_parameters=sum(p.numel() for p in model.parameters()))
    rng = np.random.default_rng(a.seed)
    history = []; start = 0; best = -float('inf'); updates = 0
    if a.resume:
        old = torch.load(root/'last.pt', map_location=a.device, weights_only=False)
        if old['protocol'] != settings:
            raise ValueError('Resume protocol/source/data/batch size changed')
        model.load_state_dict(old['model'], strict=True); opt.load_state_dict(old['optimizer'])
        start, updates, best, history = old['epoch'], old['updates'], old['best'], old['history']
        rng.bit_generator.state = old['rng']; torch.set_rng_state(old['torch_rng'].cpu())
    atomic_json(root/'protocol.json', signed(settings))
    for epoch in range(start+1, a.epochs+1):
        model.train(); order = rng.permutation(len(tr['entries'])); losses = []
        opt.zero_grad(set_to_none=True)
        # One optimizer step processes exactly one effective group. Physical
        # batch B is a *single* vectorised forward/backward; accumulation only
        # applies when the explicit accumulate flag exceeds 1.
        for start_idx in range(0, len(order), effective_batch):
            group = order[start_idx:start_idx+effective_batch]
            for micro_start in range(0, len(group), batch_size):
                micro = group[micro_start:micro_start+batch_size]
                frames = [read_frame(tr['entries'][int(j)]) for j in micro]
                batch = collate_frames(frames)
                logits, field = predict(model, a.mode, batch, a.device)
                count = torch.as_tensor(batch['quality_count'], dtype=torch.long, device=a.device)
                loss, _ = quality_loss(logits, count)
                if a.mode in ('h2', 'h2_stale'):
                    encoded, scene = field
                    xyz = torch.as_tensor(batch['target_xyz'], dtype=torch.float32, device=a.device)
                    raw = model.field.query(encoded, scene, xyz)
                    targets = {name: torch.as_tensor(batch['target_'+name], device=a.device)
                               for name in ('udf', 'projective', 'state', 'udf_mask', 'projective_mask')}
                    geo, _ = geometry_loss(raw, targets)
                    loss = loss + a.geometry_weight * geo
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                # Final partial group is image-weighted correctly; no dilution
                # from smaller last minibatches or extra empty padding.
                (loss * (len(micro)/len(group))).backward()
                losses.extend([float(loss.detach())] * len(micro))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step(); opt.zero_grad(set_to_none=True); updates += 1
        val = evaluate_index(model, a.mode, va, a.device, batch_size=val_batch_size)
        gain = val['summary']['gain']; improved = gain > best+1e-12
        if improved: best = gain
        history.append(dict(epoch=epoch, updates=updates, training_loss=float(np.mean(losses)), validation=val['summary']))
        payload = dict(schema=SCHEMA, protocol=settings, model=model.state_dict(), optimizer=opt.state_dict(),
                       epoch=epoch, updates=updates, best=best, history=history,
                       rng=rng.bit_generator.state, torch_rng=torch.get_rng_state())
        tmp = root/'last.pt.tmp'; torch.save(payload, tmp); tmp.replace(root/'last.pt')
        if improved:
            torch.save(payload, root/'best.pt'); atomic_json(root/'selected_validation.json', signed(val))
        atomic_json(root/'history.json', signed(dict(history=history)))
        print(f'[train {a.mode}] epoch={epoch} B={batch_size} accumulate={accumulation} '
              f'gain={gain:.6f} regret={val["summary"]["regret"]:.6f}', flush=True)
    selected = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
    atomic_json(root/'selection.json', signed(dict(schema=SCHEMA, mode=a.mode, admitted=best>0,
            best_epoch=selected['epoch'], selected_gain=best, checkpoint_sha256=sha256(root/'best.pt'),
            protocol_sha256=digest(settings), note='Validation selection, not confirmatory test; no geometry/quality oracle at inference')))


def load_model(path, device='cpu', allow_smoke=False):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('schema') != SCHEMA: raise ValueError('Wrong checkpoint schema')
    p = ck['protocol']; cfg = ModelConfig(**p['config'])
    if p['smoke'] and not allow_smoke: raise ValueError('Smoke head forbidden in formal inference')
    model = (LocalControl(p['local_feature_dim'], cfg.hidden) if p['mode'] == 'local' else
             PredictedGeometryControl(cfg) if p['mode'] == 'predicted' else FieldGraspModel(cfg))
    model.load_state_dict(ck['model'], strict=True)
    return model.to(device).eval(), p


def diagnose(a):
    idx = read_index(a.index, 'validation', a.smoke)
    model, p = load_model(a.weight, a.device, a.smoke)
    if p['val_fingerprint'] != idx['fingerprint']: raise ValueError('Wrong held-out validation cache')
    results = {}
    interventions = ('aligned',) if p['mode'] in ('local', 'predicted') else ('aligned', 'appearance_mean', 'appearance_shuffle', 'stale')
    for intervention in interventions:
        results[intervention] = evaluate_index(model, p['mode'], idx, a.device, intervention, batch_size=int(getattr(a, 'batch_size', 1)))
    contrasts={}
    for name, result in results.items():
        if name=='aligned': continue
        by_scene={}
        for base, changed in zip(results['aligned']['rows'],result['rows']):
            if base['row']!=changed['row']: raise ValueError('Diagnostic row mismatch')
            r=base['row'];by_scene.setdefault(r['scene'],{}).setdefault(r['domain'],[]).append(base['metrics']['selected_q']-changed['metrics']['selected_q'])
        values=np.array([np.mean([np.mean(v) for v in domains.values()]) for domains in by_scene.values()])
        rng=np.random.default_rng(0); boot=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(1)
        contrasts['aligned-'+name]=dict(mean=float(values.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n_scenes=len(values))
    if Path(a.output).exists(): raise FileExistsError(a.output)
    atomic_json(a.output, signed(dict(schema=SCHEMA, weight_sha256=sha256(a.weight), protocol=p, results=results, paired_scene_contrasts=contrasts,
        note='Interventions may be OOD; training h2_stale is the independent matched-access control. Metrics are not official AP.')))