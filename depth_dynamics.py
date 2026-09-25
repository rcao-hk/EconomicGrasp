"""State-safe primitives for the CVA metric-depth dynamics experiment.

This module deliberately imports no model, dataset, or project argument parser.
Depths are metres. Missing gradients stay missing until after connectivity has
been recorded; a zero loss may still have a connected (zero) derivative.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def capture_rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": [x.clone() for x in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


@contextmanager
def preserve_diagnostic_state(model, optimizer=None,
                              extra_attributes=("is_training", "_vis_iter", "_debug_iter"),
                              preserve_grads=True):
    """Restore RNG, buffers, modes, explicit selection flags and existing grads.

    Use around the *entire forward and audit*. Parameter updates are not allowed
    inside this guard; use snapshot_state/restore_snapshot for counterfactual
    optimizer steps. Data-loader generator/iterator state is owned by the caller.
    The module's ``training`` flags are restored individually, including mixed
    frozen/eval submodules; calling model.train(old_flag) would lose that state.
    """
    rng = capture_rng_state()
    modules = list(model.modules())
    modes = [(m, m.training) for m in modules]
    attributes = [(m, key, copy.deepcopy(getattr(m, key))) for m in modules
                  for key in extra_attributes if hasattr(m, key)]
    buffers = [(m, key, None if value is None else value.detach().clone())
               for m in modules for key, value in m._buffers.items()]
    grads = [(p, p.grad, None if p.grad is None else p.grad.detach().clone())
             for p in model.parameters()] if preserve_grads else []
    opt_state = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    try:
        yield
    finally:
        with torch.no_grad():
            for m, key, value in buffers:
                current = m._buffers.get(key)
                if value is None:
                    m._buffers[key] = None
                elif current is not None and current.shape == value.shape:
                    current.copy_(value)
                else:
                    m._buffers[key] = value
            for p, original, value in grads:
                p.grad = original
                if original is not None:
                    original.copy_(value)
        for m, mode in modes:
            m.training = mode
        for m, key, value in attributes:
            setattr(m, key, value)
        if optimizer is not None:
            optimizer.load_state_dict(opt_state)
        restore_rng_state(rng)


def _cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(v) for v in value)
    return copy.deepcopy(value)


def snapshot_state(model, optimizer=None, scheduler=None, scaler=None, **metadata):
    """Independent CPU snapshot; caller must include loader/sampler position."""
    result = {
        "model_state_dict": _cpu_clone(model.state_dict()),
        "rng_state": capture_rng_state(), "metadata": _cpu_clone(metadata),
        "module_modes": {name: m.training for name, m in model.named_modules()},
        "module_flags": {name: {key: copy.deepcopy(getattr(m, key))
                                for key in ("is_training", "_vis_iter", "_debug_iter")
                                if hasattr(m, key)}
                         for name, m in model.named_modules()},
    }
    for name, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        result[name + "_state_dict"] = None if obj is None else _cpu_clone(obj.state_dict())
    return result


def restore_snapshot(state, model, optimizer=None, scheduler=None, scaler=None):
    model.load_state_dict(state["model_state_dict"], strict=True)
    for name, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if obj is not None and state.get(name + "_state_dict") is not None:
            obj.load_state_dict(state[name + "_state_dict"])
    for name, module in model.named_modules():
        if name in state.get("module_modes", {}):
            module.training = state["module_modes"][name]
        for key, value in state.get("module_flags", {}).get(name, {}).items():
            setattr(module, key, copy.deepcopy(value))
    restore_rng_state(state["rng_state"])


def parameter_groups(model):
    """Disjoint semantic groups; includes frozen tensors for contract/audit."""
    groups = OrderedDict()
    for name, p in model.named_parameters():
        clean = name.removeprefix("module.")
        if clean.startswith("depth_net.depthnet.pretrained."):
            group = "dino"
        elif clean.startswith("depth_net.pose_aware_adapter."):
            group = "depth_film"
        elif clean.startswith("depth_net.depthnet.depth_head."):
            tail = clean[len("depth_net.depthnet.depth_head."):]
            group = "depth_dpt." + tail.split(".")[0]
        elif clean.startswith("depth_net."):
            group = "depth_other"
        elif clean.startswith("proposal_head."):
            group = "proposal"
        elif clean.startswith("spatial_enhancer."):
            group = "gse"
        elif clean.startswith("view."):
            group = "view"
        elif clean.startswith("kview_grasp_module.decoder."):
            group = "head." + clean[len("kview_grasp_module.decoder."):].split(".")[0]
        elif "cdf_head" in clean or "width_head" in clean:
            group = "head"
        elif clean.startswith(("kview_grasp_module.", "cva.", "kview_transformer.", "grasp_head.")):
            group = "cva"
        else:
            group = "other." + clean.split(".")[0]
        groups.setdefault(group, []).append((name, p))
    return groups


def depth_parameter_groups(model):
    return {name: pairs for name, pairs in parameter_groups(model).items()
            if name.startswith("depth_")}


def parameter_contract(model, optimizer=None):
    opt = {}
    if optimizer is not None:
        for i, group in enumerate(optimizer.param_groups):
            for p in group["params"]:
                opt.setdefault(id(p), []).append({"group": i, "lr": group["lr"],
                                                  "weight_decay": group.get("weight_decay", 0)})
    groups = {}
    for name, members in parameter_groups(model).items():
        groups[name] = {
            "numel": sum(p.numel() for _, p in members),
            "trainable_numel": sum(p.numel() for _, p in members if p.requires_grad),
            "parameters": [{"name": n, "shape": list(p.shape), "numel": p.numel(),
                            "requires_grad": p.requires_grad,
                            "optimizer": opt.get(id(p), [])} for n, p in members],
        }
    known = {id(p) for p in model.parameters()}
    return {
        "groups": groups,
        "trainable_missing_optimizer": [n for n, p in model.named_parameters()
                                        if p.requires_grad and id(p) not in opt]
        if optimizer is not None else None,
        "optimizer_unknown_parameter_count": len(set(opt) - known),
        "optimizer_duplicate_parameter_count": sum(len(v) - 1 for v in opt.values()),
        "modules": {n: {"training": m.training,
                         **({"is_training": bool(m.is_training)} if hasattr(m, "is_training") else {})}
                    for n, m in model.named_modules()},
        "buffers": [{"name": n, "shape": list(b.shape), "dtype": str(b.dtype)}
                    for n, b in model.named_buffers()],
    }


def _gradients(loss, members):
    active = [(n, p) for n, p in members if p.requires_grad]
    result = {n: None for n, _ in members}
    if active and torch.is_tensor(loss) and loss.requires_grad:
        if loss.numel() != 1:
            raise ValueError("Audit losses must be scalar; apply the actual training reduction first")
        values = torch.autograd.grad(loss, [p for _, p in active], retain_graph=True,
                                     allow_unused=True, materialize_grads=False)
        result.update((n, None if g is None else g.detach()) for (n, _), g in zip(active, values))
    return result


def _gradient_summary(members, gradients):
    states = {s: 0 for s in ("not_requires_grad", "unused/None", "connected_zero",
                             "connected_nonzero", "nonfinite")}
    connected = 0
    nonzero = 0
    square = 0.0
    maximum = 0.0
    for name, p in members:
        g = gradients[name]
        if not p.requires_grad:
            state = "not_requires_grad"
        elif g is None:
            state = "unused/None"
        else:
            g = g.coalesce().values() if g.is_sparse else g
            connected += p.numel()
            if not bool(torch.isfinite(g).all()):
                state = "nonfinite"
            else:
                nz = int(torch.count_nonzero(g).item())
                nonzero += nz
                square += float(g.double().square().sum().item())
                maximum = max(maximum, float(g.abs().max().item()) if g.numel() else 0.0)
                state = "connected_nonzero" if nz else "connected_zero"
        states[state] += p.numel()
    if states["nonfinite"]:
        state, norm, maximum = "nonfinite", None, None
    elif states["connected_nonzero"]:
        state, norm = "connected_nonzero", math.sqrt(square)
    elif states["connected_zero"]:
        state, norm = "connected_zero", 0.0
    elif states["unused/None"]:
        state, norm = "unused/None", None
    else:
        state, norm = "not_requires_grad", None
    return {"state": state, "state_numel": states, "connected_numel": connected,
            "total_numel": sum(p.numel() for _, p in members), "norm": norm,
            "max_abs": maximum, "nonzero_fraction": nonzero / connected
            if connected and not states["nonfinite"] else None}


def _cosine(members, current, reference, norm, reference_norm, epsilon):
    if norm is None or reference_norm is None or min(norm, reference_norm) <= epsilon:
        return None
    dot = 0.0
    for name, _ in members:
        g, d = current[name], reference[name]
        if g is not None and d is not None:
            if g.is_sparse or d.is_sparse:
                g, d = g.to_dense(), d.to_dense()
            dot += float((g.double() * d.double()).sum().item())
    return max(-1.0, min(1.0, dot / (norm * reference_norm)))


def audit_gradients(losses, groups, weights=None, observations=None,
                    reference_loss="depth", cosine_epsilon=1e-12):
    """Raw and weighted derivatives without accumulating into parameter .grad.

    groups maps group names to ``[(name, tensor)]``. observations optionally maps
    endpoint names to tensors (raw depth, metric depth, route aliases). The caller
    must preserve forward state using preserve_diagnostic_state. Returned rows
    are JSON-safe and retain group-level counts for all five connectivity states.
    ``norm_ratio_vs_depth`` uses the corresponding raw/weighted depth objective.
    """
    if reference_loss not in losses:
        raise KeyError("Reference depth loss is required for norm/cosine audit")
    groups = dict(groups)
    if observations:
        groups.update({"endpoint." + n: [("endpoint." + n, p)]
                       for n, p in observations.items() if torch.is_tensor(p)})
    all_members = OrderedDict()
    by_id = {}
    # Avoid duplicate autograd inputs while allowing aliases across groups.
    aliases = {}
    for members in groups.values():
        for name, tensor in members:
            canonical = by_id.setdefault(id(tensor), name)
            aliases[name] = canonical
            all_members[canonical] = tensor
    members = list(all_members.items())
    weights = weights or {}
    rows = []
    for scale in ("raw", "weighted"):
        ref_weight = float(weights.get(reference_loss, 1.0)) if scale == "weighted" else 1.0
        ref_loss = losses[reference_loss] * ref_weight
        reference_base = _gradients(ref_loss, members)
        reference = {n: reference_base[c] for n, c in aliases.items()}
        reference_stats = {name: _gradient_summary(items, reference) for name, items in groups.items()}
        for loss_name, loss in losses.items():
            weight = float(weights.get(loss_name, 1.0)) if scale == "weighted" else 1.0
            objective = loss * weight
            if loss_name == reference_loss:
                gradients = reference
            else:
                base = _gradients(objective, members)
                gradients = {n: base[c] for n, c in aliases.items()}
            for group_name, items in groups.items():
                summary = _gradient_summary(items, gradients)
                dn = reference_stats[group_name]["norm"]
                gn = summary["norm"]
                rows.append({"loss": loss_name, "scale": scale, "weight": weight,
                             "loss_value": float(objective.detach().item()),
                             "loss_requires_grad": bool(objective.requires_grad),
                             "group": group_name, **summary,
                             "cosine_vs_depth": _cosine(items, gradients, reference, gn, dn, cosine_epsilon),
                             "norm_ratio_vs_depth": gn / dn if gn is not None and dn is not None
                             and dn > cosine_epsilon else None})
    return rows


def _images(value, name):
    value = torch.as_tensor(value).detach().cpu()
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [B,H,W] or [B,1,H,W], got {tuple(value.shape)}")
    return value


def _quantiles(values):
    if not values.numel():
        return {"p05": None, "p50": None, "p95": None}
    q = torch.quantile(values.float(), torch.tensor([.05, .5, .95])).tolist()
    return dict(zip(("p05", "p50", "p95"), q))


def make_fixed_pairs(gt, instance=None, max_pairs=4096, seed=0,
                     min_depth=.2, max_depth=1.0, offsets=(1, 4, 16)):
    """Fixed horizontal/vertical same-instance pairs from GT only.

    Returns one [N,2] flattened-index tensor per image. Foreground instance IDs
    must be positive; if instance is omitted, pairs are merely GT-valid and this
    distinction must be recorded in the experiment contract. Uses a local RNG.
    """
    gt = _images(gt, "gt")
    instance = None if instance is None else _images(instance, "instance")
    if instance is not None and instance.shape != gt.shape:
        raise ValueError("Instance and GT shapes must match")
    gen = torch.Generator().manual_seed(int(seed))
    pairs = []
    for i, image in enumerate(gt):
        height, width = image.shape
        ids = torch.arange(height * width).reshape(height, width)
        valid = torch.isfinite(image) & (image >= min_depth) & (image <= max_depth)
        result = []
        for d in offsets:
            for axis in (0, 1):
                if d <= 0 or d >= image.shape[axis]:
                    continue
                a = (slice(None, -d), slice(None)) if axis == 0 else (slice(None), slice(None, -d))
                b = (slice(d, None), slice(None)) if axis == 0 else (slice(None), slice(d, None))
                mask = valid[a] & valid[b]
                if instance is not None:
                    mask &= (instance[i][a] > 0) & (instance[i][a] == instance[i][b])
                result.append(torch.stack((ids[a][mask], ids[b][mask]), dim=1))
        result = torch.cat(result) if result else torch.empty((0, 2), dtype=torch.long)
        if len(result) > max_pairs:
            result = result[torch.randperm(len(result), generator=gen)[:max_pairs]]
        pairs.append(result)
    return pairs


def depth_metrics(pred, gt, foreground=None, raw=None, pairs=None,
                  min_depth=.2, max_depth=1.0, contrast_epsilon=1e-6):
    """Per-image metrics on fixed GT masks; no pooled-image std is reported."""
    pred, gt = _images(pred, "pred").float(), _images(gt, "gt").float()
    if pred.shape != gt.shape:
        raise ValueError("Prediction and GT must be aligned before computing depth metrics")
    foreground = None if foreground is None else _images(foreground, "foreground").bool()
    raw = None if raw is None else _images(raw, "raw").float()
    if foreground is not None and foreground.shape != gt.shape:
        raise ValueError("Foreground and GT must be aligned")
    if raw is not None and raw.shape != pred.shape:
        raise ValueError("Raw depth and metric depth must be aligned")
    if pairs is not None and len(pairs) != len(pred):
        raise ValueError("One fixed pair index tensor is required per image")
    rows = []
    for i in range(len(pred)):
        valid = torch.isfinite(gt[i]) & (gt[i] >= min_depth) & (gt[i] <= max_depth)
        regions = {"valid": valid}
        if foreground is not None:
            regions.update(foreground=valid & foreground[i], background=valid & ~foreground[i])
        row = {"image_index": i, "pixels": pred[i].numel(), "gt_valid_count": int(valid.sum()),
               "pred_nonfinite_count": int((~torch.isfinite(pred[i])).sum()), "regions": {}}
        for region, mask in regions.items():
            fixed_count = int(mask.sum())
            selected = mask & torch.isfinite(pred[i])
            p, g = pred[i][selected], gt[i][selected]
            pq, gq = _quantiles(p), _quantiles(g)
            ps = float(p.std(unbiased=False)) if len(p) else None
            gs = float(g.std(unbiased=False)) if len(g) else None
            row["regions"][region] = {
                "count": fixed_count, "finite_count": int(selected.sum()),
                "mae": float((p-g).abs().mean()) if len(p) else None,
                "bias": float((p-g).mean()) if len(p) else None,
                "pred_mean": float(p.mean()) if len(p) else None,
                "gt_mean": float(g.mean()) if len(g) else None,
                "pred_std": ps, "gt_std": gs, "pred_quantiles": pq, "gt_quantiles": gq,
                "std_ratio": ps / gs if gs is not None and gs > contrast_epsilon else None,
                "span_ratio": (pq["p95"]-pq["p05"])/(gq["p95"]-gq["p05"])
                if len(g) and gq["p95"]-gq["p05"] > contrast_epsilon else None,
                "below_min_fraction": float((p < min_depth).float().mean()) if len(p) else None,
                "above_max_fraction": float((p > max_depth).float().mean()) if len(p) else None,
            }
        if raw is not None:
            r = raw[i][valid & torch.isfinite(raw[i])]
            sigmoid = r.sigmoid()
            deriv = sigmoid * (1-sigmoid)
            row["raw"] = {**_quantiles(r), "finite_count": len(r),
                          "nonfinite_count": int((valid & ~torch.isfinite(raw[i])).sum()),
                          "sigmoid_derivative_mean": float(deriv.mean()) if len(r) else None,
                          "sigmoid_derivative_quantiles": _quantiles(deriv),
                          "sigmoid_extreme_fraction": float(((sigmoid < .01) | (sigmoid > .99)).float().mean())
                          if len(r) else None}
        if pairs is not None:
            pair = torch.as_tensor(pairs[i], dtype=torch.long).cpu()
            pflat, gflat = pred[i].flatten(), gt[i].flatten()
            if pair.ndim != 2 or pair.shape[1] != 2:
                raise ValueError("Each fixed pair tensor must have shape [N,2]")
            finite = torch.isfinite(pflat[pair]).all(dim=1)
            legal = valid.flatten()[pair].all(dim=1)
            if not bool(legal.all()):
                raise ValueError("Fixed pairs must stay inside the fixed GT-valid region")
            a, b = pair[finite].unbind(dim=1)
            dp, dg = pflat[b]-pflat[a], gflat[b]-gflat[a]
            denom = float(dg.double().square().sum())
            gt_rms = float(dg.square().mean().sqrt()) if len(dg) else None
            pred_rms = float(dp.square().mean().sqrt()) if len(dp) else None
            dc, pc = dg-dg.mean(), dp-dp.mean()
            cnorm = float(dc.norm() * pc.norm())
            row["local"] = {
                "pair_count": len(pair), "finite_count": len(dp),
                "gt_rms": gt_rms, "pred_rms": pred_rms,
                "difference_mae": float((dp-dg).abs().mean()) if len(dp) else None,
                "slope": float((dp.double()*dg.double()).sum()) / denom
                if len(dg) and gt_rms > contrast_epsilon else None,
                "correlation": float((dc*pc).sum())/cnorm if cnorm > contrast_epsilon else None,
                "contrast_ratio": pred_rms/gt_rms if len(dg) and gt_rms > contrast_epsilon else None,
            }
        rows.append(row)
    return rows


def output_directional_derivatives(losses, pred, gt, foreground=None, min_depth=.2, max_depth=1.):
    """dL/dmu and dL/dalpha for per-image fixed-region metric-depth directions.

    These are local output-space derivatives on the current autograd graph, not
    network-update or finite-difference evidence. Discrete targets are those of
    this forward; the caller separately checks frozen-branch finite differences.
    """
    if pred.ndim == 4 and pred.shape[1] == 1:
        pimages = pred[:, 0]
    else:
        pimages = pred
    g = gt[:, 0] if gt.ndim == 4 and gt.shape[1] == 1 else gt
    mask = torch.isfinite(g) & (g >= min_depth) & (g <= max_depth)
    if foreground is not None:
        fg = foreground[:, 0] if foreground.ndim == 4 else foreground
        mask &= fg.bool()
    if pimages.shape != g.shape:
        raise ValueError("Directional probe expects aligned [B,H,W] depth and GT")
    rows = []
    for name, loss in losses.items():
        gradients = _gradients(loss, [("prediction", pred)])["prediction"]
        for i in range(len(pimages)):
            selected = mask[i]
            values = pimages[i].detach()[selected]
            row = {"loss": name, "image_index": i, "valid_count": len(values),
                   "d_loss_d_mu": None, "d_loss_d_alpha": None,
                   "state": "unused/None" if pred.requires_grad else "not_requires_grad"}
            if gradients is not None and len(values):
                grad = (gradients[:, 0] if gradients.ndim == 4 else gradients)[i][selected]
                row.update(d_loss_d_mu=float(grad.sum()),
                           d_loss_d_alpha=float((grad * (values-values.mean())).sum()),
                           state="nonfinite" if not bool(torch.isfinite(grad).all()) else
                           ("connected_nonzero" if bool(torch.count_nonzero(grad)) else "connected_zero"))
            rows.append(row)
    return rows


def json_safe(value):
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(data), sort_keys=True, allow_nan=False) + "\n")


def append_csv(path, rows, fieldnames=None):
    """Append a homogeneous table, rejecting schema drift between writes."""
    if isinstance(rows, dict):
        rows = [rows]
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames or rows[0])
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="", encoding="utf-8") as handle:
            if next(csv.reader(handle)) != fieldnames:
                raise ValueError(f"CSV schema changed for {path}")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(json_safe(v), sort_keys=True) if isinstance(v, (dict, list, tuple))
                             else json_safe(v) for k, v in row.items()})


def sha256_file(path, block_size=8 * 1024 * 1024):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            result.update(block)
    return result.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()
