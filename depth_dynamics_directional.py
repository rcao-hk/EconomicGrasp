"""Local output-space replay for CVA depth dynamics (single-process only).

The frozen function fixes image seed/view identities, both label passes and
attention-valid masks, while physical XYZ, sampling grids/radii and support
depth remain functions of the perturbed metric depth. All E/Q/C boundaries are
opened for this probe. Native finite differences deliberately recompute the
discrete decisions and are reported as secants, never as backward checks.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import math

import torch
import torch.nn.functional as F

from depth_dynamics import (
    append_csv, capture_rng_state, preserve_diagnostic_state, restore_rng_state,
)


LABEL_KEYS = (
    "batch_grasp_point", "batch_grasp_view_graspness", "batch_valid_mask",
    "batch_grasp_cdf_bins_angle_depth", "batch_grasp_cdf_valid_mask",
    "batch_grasp_cdf_pos_mask", "batch_grasp_width_angle_depth",
    "batch_grasp_width_valid_mask_angle_depth", "batch_grasp_cdf_thresholds",
    "C: Valid Points",
)
LOSS_KEYS = {
    "depth": "B: DepthReg Loss", "objectness": "B: Objectness Loss",
    "graspness": "B: Graspness Loss", "view": "B: View Loss",
    "cdf": "B: CDF Loss", "width": "B: Width Loss",
}
WEIGHT_KEYS = {
    "depth": "depth_prob_loss_weight", "objectness": "objectness_loss_weight",
    "graspness": "graspness_loss_weight", "view": "view_loss_weight",
    "cdf": "score_loss_weight", "width": "width_loss_weight",
}


def _detach(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _detach(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_detach(v) for v in value)
    return value


@contextmanager
def _replace_attribute(obj, name, replacement):
    # Restoring an inherited bound method via setattr leaves an unnecessary
    # instance override. Preserve whether the instance had its own attribute.
    own = name in vars(obj)
    previous = vars(obj).get(name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        if own:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


@contextmanager
def _replace_global(namespace, name, replacement):
    previous = namespace[name]
    namespace[name] = replacement
    try:
        yield
    finally:
        namespace[name] = previous


def _losses(end_points, loss_fn, weights):
    result = loss_fn(end_points)
    if isinstance(result, tuple):
        _, end_points = result
    values = {name: end_points[key] for name, key in LOSS_KEYS.items()}
    # Keep every component in raw and actual weighted units.
    losses = {"raw_" + name: value for name, value in values.items()}
    losses.update({"weighted_" + name: value * weights[name]
                   for name, value in values.items()})
    losses["weighted_task"] = sum(values[name] * weights[name]
                                   for name in values if name != "depth")
    return losses, end_points


def _identity_stats(trace, reference):
    seed_same = trace["seed_indices"] == reference["seed_indices"]
    query_same = ((trace["query_indices"] == reference["query_indices"])
                  & (trace["query_views"] == reference["query_views"]))
    nn_same = trace["labels"][-1]["nn_indices"] == reference["labels"][-1]["nn_indices"]
    valid_now = trace["labels"][-1]["payload"]["batch_grasp_cdf_valid_mask"]
    valid_ref = reference["labels"][-1]["payload"]["batch_grasp_cdf_valid_mask"]
    xyz = trace["query_xyz"]
    displacement = (xyz - reference["query_xyz"]).norm(dim=-1)
    old_match_distance = (xyz - reference["labels"][-1]["payload"]["batch_grasp_point"]).norm(dim=-1)
    native_attention = torch.cat([v.reshape(-1) for v in trace["native_attention_valid"]])
    ref_attention = torch.cat([v.reshape(-1) for v in reference["native_attention_valid"]])
    return {
        "seed_switch_rate_aligned_slots": float((~seed_same).float().mean()),
        "query_identity_switch_rate_aligned_slots": float((~query_same).float().mean()),
        "nn_switch_rate_aligned_slots": float((~nn_same).float().mean()),
        "nn_switch_rate_same_identity": float((~nn_same)[query_same].float().mean())
        if bool(query_same.any()) else None,
        "same_identity_query_count": int(query_same.sum()),
        "cdf_mask_switch_rate_aligned_slots": float((valid_now != valid_ref).float().mean()),
        "attention_native_mask_switch_rate": float((native_attention != ref_attention).float().mean()),
        "query_displacement_mean_m": float(displacement.mean()),
        "query_displacement_max_m": float(displacement.max()),
        "distance_to_frozen_match_mean_m": float(old_match_distance.mean()),
        "distance_to_frozen_match_max_m": float(old_match_distance.max()),
    }


def _trace_summary(trace):
    """Drop graph-bearing model endpoints, retaining only discrete replay data."""
    return {k: _detach(v) for k, v in trace.items()}


def _check_frozen_identity(trace, reference):
    for key in ("seed_indices", "query_indices", "query_views"):
        if not torch.equal(trace[key], reference[key]):
            raise RuntimeError(f"Frozen directional replay changed {key}")
    if len(trace["labels"]) != len(reference["labels"]) or len(trace["labels"]) != 2:
        raise RuntimeError("Expected exactly two CVA CDF label passes")
    for actual, expected in zip(trace["labels"], reference["labels"]):
        for key in LABEL_KEYS:
            if not torch.equal(actual["payload"][key], expected["payload"][key]):
                raise RuntimeError(f"Frozen replay changed label/target {key}")
    for actual, expected in zip(trace["attention_valid"], reference["attention_valid"]):
        if not torch.equal(actual, expected):
            raise RuntimeError("Frozen replay changed attention-valid mask")


def run_directional_probe(
    model, batch, loss_fn, *, loss_weights=None, batch_id=0, output_csv=None,
    relative_eps=(1e-4, 1e-3, 1e-2), image_indices=(0,), foreground=None,
):
    """Return raw/weighted loss directions and frozen/native central secants.

    Default cost: capture + frozen differentiable check + 24 perturbation
    forwards, for one image's fixed GT-valid foreground (or GT-valid fallback).
    Each evaluation starts from the same RNG and module buffers/modes. Training
    dropout remains enabled if enabled on entry; selection methods still run to
    preserve random-number consumption before their identities are overridden.
    No optimizer is touched. Run outside concurrent forwards / DDP collectives.

    ``loss_fn`` is the existing get_loss_cdf-style callable. ``loss_weights``
    uses depth/objectness/graspness/view/cdf/width keys; omitted weights are read
    from that function's actual cfgs. Only image-FPS RGB-only CDF is supported.
    """
    if hasattr(model, "module"):
        raise ValueError("Directional replay requires the unwrapped single-process model")
    if (model.seed_selection_mode != "image_fps" or model.geometry_depth_source != "pred"
            or model.use_obs_depth or model.use_gt_xyz_for_train or not model.use_cdf):
        raise ValueError("Directional replay requires RGB-only predicted-depth image-FPS CDF")
    if loss_weights is None:
        cfg = loss_fn.__globals__.get("cfgs")
        if cfg is None:
            raise ValueError("Pass actual loss_weights when loss_fn has no cfgs")
        loss_weights = {name: float(getattr(cfg, key)) for name, key in WEIGHT_KEYS.items()}
    else:
        loss_weights = {name: float(loss_weights[name]) for name in LOSS_KEYS}
    if any(not math.isfinite(x) or x <= 0 for x in relative_eps):
        raise ValueError("Finite-difference relative_eps must be finite and positive")

    forward_globals = model.forward.__func__.__globals__
    label_fn = forward_globals["process_grasp_labels_cdf_width"]
    label_globals = label_fn.__globals__
    original_knn = label_globals["knn_points"]
    view = model.view
    selector = model.kview_grasp_module.selector
    group = model.kview_grasp_module.group
    view_select = view._select_top_view_inds
    query_select = selector._select_view_indices
    make_grid = group._make_view_conditioned_grid
    old_routes = model.get_depth_grad_routes()
    # Preserve each real flag, including independently replaced config objects.
    route_objects = [(model.spatial_enhancer, "detach_depth_grad"),
                     (model, "detach_seed_xyz_grad"),
                     (model.kview_config, "detach_depth"),
                     (model.kview_grasp_module.config, "detach_depth"),
                     (group.config, "detach_depth"), (model, "depth_grad_routes")]
    old_route_values = [(obj, name, getattr(obj, name)) for obj, name in route_objects]
    rng = capture_rng_state()
    rows = []

    def evaluate(z_override=None, reference=None, need_grad=False):
        trace = {"labels": [], "attention_valid": [], "native_attention_valid": []}
        base_ep = dict(batch)
        # Existing oracle-view override changes scores; it is prohibited here.
        base_ep.pop("oracle_view_inds_override", None)
        base_ep.pop("image_fps_seed_idx_override", None)
        base_ep["cva_force_process_grasp_labels"] = True
        base_ep["cva_compute_diagnostics"] = False
        base_ep["depth_grad_capture_routes"] = False
        if reference is not None:
            base_ep["image_fps_seed_idx_override"] = reference["seed_indices"]

        def depth_output_hook(module, args, output):
            if not isinstance(output, tuple) or len(output) != 6:
                raise RuntimeError("Unexpected RGB depth-net output contract")
            if z_override is None:
                z = output[0].detach().clone()
            else:
                z = z_override
            trace["depth"] = z
            tok = z if output[1].shape == z.shape else F.interpolate(
                z, size=output[1].shape[-2:], mode="nearest")
            return (z, tok, *tuple(_detach(x) for x in output[2:]))

        def selected_view(scores):
            native = view_select(scores)  # consumes exactly the native RNG
            trace["native_base_views"] = native.detach().clone()
            selected = reference["base_views"] if reference is not None else native
            trace["base_views"] = selected.detach().clone()
            return selected

        def selected_queries(scores, is_training, forced_view_inds=None):
            result = query_select(scores, is_training, forced_view_inds=forced_view_inds)
            ids, ranks, selected_prob, effective_k, probability = result
            trace["native_query_views_bmk"] = ids.detach().clone()
            if reference is not None:
                ids = reference["query_views_bmk"]
                ranks = reference["query_ranks_bmk"]
                if ids.shape[-1] != effective_k:
                    raise RuntimeError("Frozen query multiplicity changed")
                selected_prob = torch.gather(probability, -1, ids)
            trace["query_views_bmk"] = ids.detach().clone()
            trace["query_ranks_bmk"] = ranks.detach().clone()
            return ids, ranks, selected_prob, effective_k, probability

        def labels(ep):
            call_index = len(trace["labels"])
            if reference is not None:
                item = reference["labels"][call_index]
                payload = _detach(item["payload"])
                ep.update(payload)
                trace["labels"].append({"payload": payload,
                                         "rotation": item["rotation"],
                                         "nn_indices": item["nn_indices"]})
                return item["rotation"], ep
            centers = ep["xyz_graspable"]
            nn = {}
            pointers = {centers[i].data_ptr(): i for i in range(len(centers))}

            def capture_knn(x, y, *args, **kwargs):
                result = original_knn(x, y, *args, **kwargs)
                # The other KNN calls align fixed canonical views, not queries.
                if x.data_ptr() in pointers and x.shape == (1, centers.shape[1], 3):
                    nn[pointers[x.data_ptr()]] = result[1].reshape(-1).detach().clone()
                return result

            with _replace_global(label_globals, "knn_points", capture_knn):
                rotation, ep = label_fn(ep)
            if len(nn) != len(centers):
                raise RuntimeError("Could not capture every CDF nearest-point assignment")
            trace["labels"].append({
                "payload": {key: _detach(ep[key]) for key in LABEL_KEYS},
                "rotation": _detach(rotation),
                "nn_indices": torch.stack([nn[i] for i in range(len(centers))]),
            })
            return rotation, ep

        def grid(*args, **kwargs):
            output = list(make_grid(*args, **kwargs))
            index = len(trace["attention_valid"])
            trace["native_attention_valid"].append(output[1].detach().clone())
            if reference is not None:
                output[1] = reference["attention_valid"][index]
            trace["attention_valid"].append(output[1].detach().clone())
            return tuple(output)

        with preserve_diagnostic_state(model), ExitStack() as stack:
            restore_rng_state(rng)
            stack.enter_context(_replace_attribute(view, "_select_top_view_inds", selected_view))
            stack.enter_context(_replace_attribute(selector, "_select_view_indices", selected_queries))
            stack.enter_context(_replace_attribute(group, "_make_view_conditioned_grid", grid))
            stack.enter_context(_replace_attribute(view, "_runtime_oracle_view_inds_override", None))
            stack.enter_context(_replace_global(forward_globals, "process_grasp_labels_cdf_width", labels))
            hook = model.depth_net.register_forward_hook(depth_output_hook)
            stack.callback(hook.remove)
            with torch.enable_grad() if need_grad else torch.no_grad():
                ep = model(base_ep)
                losses, ep = _losses(ep, loss_fn, loss_weights)
                trace.update(
                    seed_indices=ep["kview_base_token_sel_idx"].detach().clone(),
                    query_indices=ep["token_sel_idx"].detach().clone(),
                    query_views=ep["grasp_top_view_inds"].detach().clone(),
                    query_xyz=ep["xyz_graspable"].detach().clone(),
                )
                if len(trace["labels"]) != 2:
                    raise RuntimeError("Expected two CDF label passes in directional replay")
                if reference is not None:
                    _check_frozen_identity(trace, reference)
                values = {name: float(loss.detach()) for name, loss in losses.items()}
                gradients = {}
                if need_grad:
                    # Compute each distinct raw gradient once; weighted values
                    # are exact scalar multiples, task is the weighted sum.
                    for name in LOSS_KEYS:
                        loss = losses["raw_" + name]
                        grad = None
                        if loss.requires_grad:
                            grad, = torch.autograd.grad(
                                loss, z_override, retain_graph=True,
                                allow_unused=True, materialize_grads=False)
                        gradients["raw_" + name] = None if grad is None else grad.detach()
                        gradients["weighted_" + name] = None if grad is None else grad.detach() * loss_weights[name]
                    task_parts = [gradients["weighted_" + name] for name in LOSS_KEYS
                                  if name != "depth" and gradients["weighted_" + name] is not None]
                    gradients["weighted_task"] = sum(task_parts) if task_parts else None
            trace = _trace_summary(trace)
        return values, gradients, trace

    try:
        model.set_depth_grad_routes("all")
        baseline_losses, _, reference = evaluate()
        z0 = reference["depth"]
        leaf = z0.clone().requires_grad_(True)
        frozen_losses, gradients, frozen_trace = evaluate(leaf, reference, need_grad=True)
        # An identity-control failure is an implementation error, not a result.
        for key in baseline_losses:
            if not math.isclose(baseline_losses[key], frozen_losses[key], rel_tol=1e-6, abs_tol=1e-7):
                raise RuntimeError(f"Frozen replay identity control failed for {key}: "
                                   f"{baseline_losses[key]} vs {frozen_losses[key]}")
        gt = batch["gt_depth_m"]
        if gt.ndim == 3:
            gt = gt.unsqueeze(1)
        if gt.shape != z0.shape:
            raise ValueError("Directional GT and metric output must be aligned BCHW")
        gt_valid = torch.isfinite(gt) & (gt >= 0.2) & (gt <= 1.0)
        if foreground is None and "objectness_label_tok" in batch:
            foreground = batch["objectness_label_tok"].reshape_as(z0) == 1
        if foreground is not None:
            foreground = foreground.reshape_as(z0).bool()
        for image_index in image_indices:
            if not 0 <= image_index < len(z0):
                raise ValueError(f"Directional image index {image_index} is out of range")
            region = gt_valid[image_index].clone()
            region_name = "gt_valid"
            if foreground is not None and bool((region & foreground[image_index]).any()):
                region &= foreground[image_index]
                region_name = "gt_valid_foreground"
            if not bool(region.any()):
                raise ValueError("No valid pixels in directional probe image")
            values = z0[image_index][region]
            if not bool(torch.isfinite(values).all()):
                raise ValueError("Nonfinite prediction inside fixed directional region")
            mean = float(values.double().mean())
            mask = torch.zeros_like(z0)
            mask[image_index][region] = 1.0
            directions = {"mu": mask, "alpha": mask * (z0 - mean)}
            for coordinate, direction in directions.items():
                exact = {}
                states = {}
                for name, grad in gradients.items():
                    if grad is None:
                        exact[name], states[name] = None, "unused/None"
                    elif not bool(torch.isfinite(grad).all()):
                        exact[name], states[name] = None, "nonfinite"
                    else:
                        exact[name] = float((grad.double() * direction.double()).sum())
                        states[name] = "connected_nonzero" if bool(torch.count_nonzero(grad)) else "connected_zero"
                for relative_step in relative_eps:
                    step = relative_step * max(abs(mean), 1e-3) if coordinate == "mu" else relative_step
                    for protocol in ("frozen_continuous", "native_recompute"):
                        fixed = reference if protocol == "frozen_continuous" else None
                        minus, _, minus_trace = evaluate(z0 - step * direction, fixed)
                        plus, _, plus_trace = evaluate(z0 + step * direction, fixed)
                        minus_stats = _identity_stats(minus_trace, reference)
                        plus_stats = _identity_stats(plus_trace, reference)
                        for name in baseline_losses:
                            secant = (plus[name] - minus[name]) / (2.0 * step)
                            derivative = exact[name]
                            # Only the frozen function's FD is a gradient check.
                            abs_error = (abs(secant - derivative)
                                         if derivative is not None and protocol == "frozen_continuous" else None)
                            rows.append({
                                "batch_id": batch_id, "image_index": image_index,
                                "route": "all", "entry_routes": old_routes,
                                "protocol": protocol, "coordinate": coordinate,
                                "region": region_name, "region_count": int(region.sum()),
                                "mu0_m": mean, "region_std_m": float(values.double().std(unbiased=False)),
                                "relative_step": relative_step, "coordinate_step": step,
                                "max_pixel_perturbation_m": float((step * direction).abs().max()),
                                "loss": name, "loss_base": baseline_losses[name],
                                "loss_minus": minus[name], "loss_plus": plus[name],
                                "autograd_fixed_branch": derivative,
                                "gradient_state": states[name], "central_secant": secant,
                                "frozen_fd_abs_error": abs_error,
                                "frozen_fd_relative_error": abs_error / max(abs(secant), abs(derivative), 1e-10)
                                if abs_error is not None else None,
                                "minus_switch_stats": minus_stats, "plus_switch_stats": plus_stats,
                                "module_training": bool(model.training),
                                "selection_is_training": bool(model.is_training),
                                "interpretation": "gradient_check_of_fixed_discrete_function"
                                if protocol == "frozen_continuous" else "native_secant_with_discrete_recomputation",
                            })
    finally:
        for obj, name, value in old_route_values:
            setattr(obj, name, value)
        restore_rng_state(rng)
    if output_csv is not None:
        append_csv(output_csv, rows)
    return rows
