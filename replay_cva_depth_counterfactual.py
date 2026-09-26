"""P4: one actual AdamW update from one full depth-dynamics checkpoint.

The only intervention subtracts a selected current weighted task derivative
from depth-exclusive DPT/FiLM parameter gradients. Historical Adam moments are
retained. Four branches share one forward/backward realization. This measures a
local conditional update, not additive loss-specific Adam updates or a method.
No model/trainer implementation is modified. --help needs no CUDA dependencies.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="One full dynamics step_*.pt snapshot")
    parser.add_argument("--init_checkpoint", required=True, help="Original initialization, hash must match snapshot")
    parser.add_argument("--output", required=True, help="New, empty output directory")
    parser.add_argument("--target_loss", choices=("view", "task"), default="view",
                        help="task means the explicit sum of all weighted non-depth losses")
    parser.add_argument("--probe_modes", choices=("eval", "both"), default="eval")
    return parser.parse_args(argv)


def digest(value):
    """Hash actual tensor/array bytes and structure, without pickle alias effects."""
    import numpy as np
    import torch
    result = hashlib.sha256()

    def visit(item):
        result.update(type(item).__name__.encode() + b":")
        if torch.is_tensor(item):
            item = item.detach().cpu().contiguous()
            result.update(str((item.dtype, tuple(item.shape))).encode())
            result.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            result.update(str((item.dtype, item.shape)).encode())
            result.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for entry in item:
                visit(entry)
        else:
            result.update(repr(item).encode())
        result.update(b";")

    visit(value)
    return result.hexdigest()


def tensor_stats(value):
    import torch
    if value is None:
        return {"state": "unused/None", "numel": 0, "l2": None}
    value = value.detach().double()
    finite = torch.isfinite(value)
    good = value[finite]
    count = int(torch.count_nonzero(good))
    return {"state": "nonfinite" if not bool(finite.all()) else
            "connected_nonzero" if count else "connected_zero",
            "numel": value.numel(), "nonfinite_count": int((~finite).sum()),
            "nonzero_count": count, "l2": float(good.norm()),
            "max_abs": float(good.abs().max()) if good.numel() else 0.0,
            "mean": float(good.mean()) if good.numel() else None}


def capture_runtime(model):
    return {
        "buffers": {name: value.detach().cpu().clone() for name, value in model.named_buffers()},
        "modes": {name: module.training for name, module in model.named_modules()},
        "flags": {name: {key: copy.deepcopy(getattr(module, key))
                          for key in ("is_training", "_vis_iter", "_debug_iter", "_forward_count", "_step")
                          if hasattr(module, key)} for name, module in model.named_modules()},
    }


def restore_runtime(model, runtime):
    import torch
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(runtime["buffers"][name].to(value.device))
    for name, module in model.named_modules():
        module.training = runtime["modes"][name]
        for key, value in runtime["flags"][name].items():
            setattr(module, key, copy.deepcopy(value))


def restore_full_state(experiment, state):
    import depth_dynamics as dd
    model, optimizer = experiment.model, experiment.optimizer
    model.load_state_dict(state["model_state_dict"], strict=True)
    # Even CUDA AdamW can retain the CPU step counter by reference.
    optimizer.load_state_dict(copy.deepcopy(state["optimizer_state_dict"]))
    optimizer.zero_grad(set_to_none=True)
    experiment.stream.load_state_dict(copy.deepcopy(state["loader"]))
    experiment.step, experiment.seen_images = state["step"], state["seen_images"]
    model.set_depth_grad_routes(",".join(key for key, enabled in state["routes"].items() if enabled) or "none")
    for name, module in model.named_modules():
        module.training = state["module_training"][name]
        if name in state["is_training"]:
            module.is_training = state["is_training"][name]
        for key, value in state.get("runtime_counters", {}).get(name, {}).items():
            setattr(module, key, copy.deepcopy(value))
    if "_replay_runtime" in state:
        restore_runtime(model, state["_replay_runtime"])
    dd.restore_rng_state(state["rng"])


def install_gradients(model, captured, removed=None):
    """Retain None vs zero and alter only names in the explicit removal scope."""
    removed = removed or {}
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(captured) != set(parameters) or not set(removed).issubset(parameters):
        raise ValueError("Captured/removal gradient names do not match trainable parameters")
    for name, parameter in parameters.items():
        total, target = captured[name], removed.get(name)
        if total is None:
            if target is not None:
                raise ValueError(f"Target is connected but total gradient is None: {name}")
            parameter.grad = None
        else:
            if total.shape != parameter.shape or (target is not None and target.shape != parameter.shape):
                raise ValueError(f"Captured/removal gradient shape mismatch: {name}")
            parameter.grad = total.to(device=parameter.device, dtype=parameter.dtype).clone()
            if target is not None:
                parameter.grad.sub_(target.to(device=parameter.device, dtype=parameter.dtype))


def apply_clip(model, max_norm, fixed_coefficient=None):
    import torch
    parameters = list(model.parameters())
    # Infinity leaves the gradients unchanged while using the identical native
    # total-norm computation and nonfinite guard for the fixed-coefficient arm.
    norm = torch.nn.utils.clip_grad_norm_(parameters,
        max_norm if fixed_coefficient is None else math.inf, error_if_nonfinite=True)
    if fixed_coefficient is None:
        coefficient = float((max_norm / (norm + 1e-6)).clamp(max=1.0))
    else:
        coefficient = float(fixed_coefficient)
        if not math.isfinite(coefficient) or not 0.0 <= coefficient <= 1.0:
            raise ValueError("Fixed clip coefficient must be finite and in [0,1]")
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)
    return float(norm), coefficient


def config_cli(saved):
    """Serialize the actual existing parser; Experiment.resume checks equality."""
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        parser = runpy.run_path(str(Path(__file__).parent / "utils/arguments.py"))["parser"]
    finally:
        sys.argv = old_argv
    owned = {"distill_stage", "use_cdf", "multi_modal", "extend_angle", "seed", "log_dir",
             "num_workers", "eval_num_workers", "resume", "checkpoint_path"}
    result = []
    for action in parser._actions:
        if action.dest not in saved or action.dest in owned:
            continue
        value = saved[action.dest]
        if value == action.default or value is None:
            continue
        option = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                result.append(option)
            else:
                raise ValueError(f"Cannot reconstruct false store_true default: {action.dest}")
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                result.append(option)
            else:
                raise ValueError(f"Cannot reconstruct true store_false default: {action.dest}")
        elif isinstance(action, argparse._StoreAction):
            result += [option, *map(str, value if isinstance(value, (list, tuple)) else [value])]
        else:
            raise ValueError(f"Unsupported saved parser action: {action.dest}")
    return result


def probe(experiment, modes):
    import torch
    import depth_dynamics as dd
    from train_cva_depth_dynamics import stable_seed
    rows = []
    for training in ([False] if modes == "eval" else [True, False]):
        with dd.preserve_diagnostic_state(experiment.model, extra_attributes=(
                "is_training", "_vis_iter", "_debug_iter", "_forward_count", "_step"), preserve_grads=False):
            experiment.model.train(training)
            for module in experiment.model.modules():
                if hasattr(module, "is_training"):
                    module.is_training = training
            experiment.original.seed_everything(stable_seed(experiment.args.seed, "probe", int(training)))
            with torch.no_grad():
                for item in experiment.probe_cache:
                    batch = experiment.original.collate_fn([item["sample"]])
                    _, endpoints = experiment.forward(batch)
                    pred = endpoints["depth_net_pred"].detach().cpu().clone()
                    raw = endpoints["depth_head_raw_pred"].detach().cpu().clone()
                    rows.append({"split": item["split"], "index": item["index"],
                        "scene": item["scene"], "frame": item["frame"],
                        "module_mode": "train" if training else "eval",
                        "metrics": dd.depth_metrics(pred, item["gt"], foreground=item["instances"] > 0,
                                                    raw=raw, pairs=item["pairs"]),
                        "pred": pred, "raw": raw})
    return rows


def probe_report(current, references):
    rows = []
    for index, row in enumerate(current):
        report = {key: value for key, value in row.items() if key not in ("pred", "raw")}
        report["prediction_deltas"] = {}
        for label, reference in references.items():
            other = reference[index]
            for key in ("split", "index", "module_mode"):
                if row[key] != other[key]:
                    raise ValueError("Fixed probe identity changed")
            report["prediction_deltas"][label] = {
                endpoint: tensor_stats(row[endpoint] - other[endpoint]) for endpoint in ("pred", "raw")}
        rows.append(report)
    return rows


def parameter_deltas(model, reference, groups):
    per_parameter, totals = {}, {}
    names_to_group = {name: group for group, pairs in groups.items() for name, _ in pairs}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        stats = tensor_stats(parameter.detach().cpu() - reference[name])
        per_parameter[name] = stats
        row = totals.setdefault(names_to_group[name], {"squared_l2": 0.0, "max_abs": 0.0, "numel": 0})
        row["squared_l2"] += stats["l2"] ** 2
        row["max_abs"] = max(row["max_abs"], stats["max_abs"])
        row["numel"] += stats["numel"]
    for row in totals.values():
        row["l2"] = math.sqrt(row.pop("squared_l2"))
    return {"groups": totals, "parameters": per_parameter}


def momentum_report(optimizer, scope):
    result = {}
    by_id = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    for name, parameter in scope:
        group, state = by_id[id(parameter)], optimizer.state.get(parameter, {})
        first = tensor_stats(state.get("exp_avg"))
        second = tensor_stats(state.get("exp_avg_sq"))
        gradient = tensor_stats(parameter.grad)
        beta1, beta2 = group["betas"]
        result[name] = {"step": float(state.get("step", 0)), "lr": group["lr"],
            "weight_decay": group["weight_decay"], "betas": [beta1, beta2],
            "exp_avg": first, "exp_avg_sq": second, "gradient": gradient,
            "retained_first_moment_component_l2": beta1 * (first["l2"] or 0.0),
            "current_gradient_first_moment_component_l2": (1-beta1) * (gradient["l2"] or 0.0),
            "retained_second_moment_component_l2": beta2 * (second["l2"] or 0.0),
            "step_will_skip_parameter": parameter.grad is None}
    return result


def run(experiment, initial, options):
    import torch
    import depth_dynamics as dd
    import train_cva_depth_dynamics as driver
    output = Path(options.output)
    model, optimizer = experiment.model, experiment.optimizer
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("Counterfactual contract requires the existing AdamW optimizer")
    if initial.get("scaler_state") is not None:
        raise ValueError("P4 is FP32 only; scaled checkpoint is unsupported")
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if any(p.dtype != torch.float32 for p in parameters.values()):
        raise ValueError("All trainable parameters must be FP32")
    scope = [(name, p) for group, pairs in experiment.groups.items()
             if group == "depth_film" or group.startswith("depth_dpt.")
             for name, p in pairs if p.requires_grad]
    if not scope:
        raise ValueError("No trainable depth-exclusive DPT/FiLM parameters")
    scope_names = {name for name, _ in scope}
    # Complete checkpoints contain persistent buffers. Capture construction-time
    # ephemeral buffers/counters once, too, and restore identically per branch.
    initial["_replay_runtime"] = capture_runtime(model)
    source_digest = digest({key: initial[key] for key in (
        "model_state_dict", "optimizer_state_dict", "rng", "loader", "routes", "_replay_runtime")})
    report = {"format_version": 1, "status": "running", "target_loss": options.target_loss,
        "source_checkpoint": {"path": str(Path(options.checkpoint).resolve()),
            "sha256": dd.sha256_file(options.checkpoint), "step": initial["step"],
            "seen_images": initial["seen_images"], "init_sha256": initial["init_sha256"],
            "stored_git": initial.get("git")},
        "actual_optimizer_updates_per_branch": 1, "resulting_step": initial["step"] + 1,
        "git": driver.git_info(), "resolved_config": vars(experiment.cfg),
        "source_arguments": initial["arguments"], "source_scheduler_state": initial["scheduler_state"],
        "probe_modes": options.probe_modes, "scope_parameters": sorted(scope_names),
        "scope_groups": [key for key in experiment.groups if key == "depth_film" or key.startswith("depth_dpt.")],
        "routes": model.get_depth_grad_routes(), "loss_weights": experiment.weights,
        "semantics": {
            "update": "one real AdamW step with retained historical moments; no additive per-loss update claim",
            "intervention": "subtract selected weighted current task derivative only in DPT/FiLM gradients",
            "common_forward": "one captured native batch, selection, assignment, masks, forward buffers, RNG and total backward",
            "normal_control": "optimizer-only duplicate replay from identical captured gradients, not independent CUDA backward reruns",
            "fixed_clip": "normal_A native global clip coefficient applied unchanged to every gradient after subtraction",
            "recomputed_clip": "native global clip recomputed after subtraction; can change non-depth updates",
            "momentum": "no moment reset or historical loss-component attribution; residual movement is expected",
            "probe": "fixed GT masks/same-instance pairs; same frame order and seeded train/eval; native task matching",
            "buffers": "post-forward buffers shared by all updates; extra pre-optimizer probe separates buffer-only effects",
            "scope_limit": "conditional single-state intervention; no training-trajectory, collapse-cause or AP conclusion"},
        "probe_manifest": experiment.contract["probe_manifest"], "branches": {}}
    module_names = {driver.__name__, dd.__name__, type(model).__module__,
                    "models.economicgrasp_bip3d", "models.economicgrasp_depth",
                    "models.grasp_spatial_enhancer", "models.kview_query_transformer",
                    "utils.label_generation", "label_generation", "dataset.cdf_label_adapter"}
    paths = {Path(__file__).resolve()} | {
        Path(module.__file__).resolve() for name, module in sys.modules.items()
        if name in module_names and getattr(module, "__file__", None)}
    report["executed_source_sha256"] = {str(path): dd.sha256_file(path) for path in sorted(paths)}
    report["environment"] = {"python": sys.version, "torch": torch.__version__,
                             "cuda": torch.version.cuda, "device": torch.cuda.get_device_name()}
    normal_parameters = normal_probe = normal_state_hash = None
    selected_maps = {}
    try:
        restore_full_state(experiment, initial)
        before = probe(experiment, options.probe_modes)
        report["before_step_probe"] = probe_report(before, {})
        restore_full_state(experiment, initial)
        model.train()
        for module in model.modules():
            if hasattr(module, "is_training"):
                module.is_training = True
        batch, indices = experiment.stream.next()
        if experiment.args.lr_schedule == "source_cosine":
            experiment.trainer.adjust_learning_rate(min(experiment.initial_epoch + experiment.stream.epoch,
                                                        experiment.cfg.max_epoch))
        scheduled_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.zero_grad(set_to_none=True)
        report["batch"] = {"indices": indices, "sha256": digest(batch),
            "loader_before": initial["loader"], "loader_after": copy.deepcopy(experiment.stream.state_dict()),
            "rng_before_forward_sha256": digest(dd.capture_rng_state())}
        report["resulting_seen_images"] = initial["seen_images"] + len(indices)
        loss, endpoints = experiment.forward(batch)
        terms = experiment.losses(endpoints)
        target = (terms["view"] * experiment.weights["view"] if options.target_loss == "view" else
                  sum(value * experiment.weights[key] for key, value in terms.items() if key != "depth"))
        report["forward"] = {"total_loss": float(loss.detach()), "weighted_target_loss": float(target.detach()),
            "raw_losses": {name: float(value.detach()) for name, value in terms.items()},
            "tensors_sha256": {name: digest(value) for name, value in experiment.equality_tensors(endpoints).items()},
            "coverage": experiment.scalar_metrics(endpoints)}
        target_gradients = (torch.autograd.grad(target, [p for _, p in scope], retain_graph=True, allow_unused=True)
                            if target.requires_grad else [None] * len(scope))
        removed = {name: None if gradient is None else gradient.detach().cpu().clone()
                   for (name, _), gradient in zip(scope, target_gradients)}
        del target_gradients
        if any(parameter.grad is not None for parameter in parameters.values()):
            raise RuntimeError("Target autograd unexpectedly accumulated parameter .grad")
        loss.backward()
        captured = {name: None if p.grad is None else p.grad.detach().cpu().clone() for name, p in parameters.items()}
        for collection in (captured, removed):
            if any(value is not None and not bool(torch.isfinite(value).all()) for value in collection.values()):
                raise FloatingPointError("Nonfinite captured total or target gradient")
        report["gradients"] = {"total_preclip_sha256": digest(captured), "target_preclip_sha256": digest(removed),
            "target_parameters": {name: tensor_stats(value) for name, value in removed.items()},
            "target_connected_parameter_count": sum(value is not None for value in removed.values())}
        post_runtime, post_rng = capture_runtime(model), dd.capture_rng_state()
        report["batch"]["rng_after_backward_sha256"] = digest(post_rng)
        report["batch"]["post_forward_runtime_sha256"] = digest(post_runtime)
        report["batch"]["scheduled_lrs"] = scheduled_lrs
        del loss, endpoints, target, terms, batch
        optimizer.zero_grad(set_to_none=True)
        pre_optimizer = probe(experiment, options.probe_modes)
        report["after_forward_pre_optimizer_probe"] = probe_report(pre_optimizer, {"before_step": before})
        coefficient = None
        for branch in ("normal_A", "normal_B", "remove_fixed_clip", "remove_recomputed_clip"):
            restore_full_state(experiment, initial)
            restore_runtime(model, post_runtime)
            dd.restore_rng_state(post_rng)
            experiment.stream.load_state_dict(copy.deepcopy(report["batch"]["loader_after"]))
            for group, lr in zip(optimizer.param_groups, scheduled_lrs):
                group["lr"] = lr
            pre_hash = {"model": digest(model.state_dict()), "optimizer": digest(optimizer.state_dict()),
                        "rng": digest(dd.capture_rng_state()), "runtime": digest(capture_runtime(model)),
                        "loader": digest(experiment.stream.state_dict())}
            install_gradients(model, captured, removed if branch.startswith("remove") else None)
            unchanged = all((p.grad is None and captured[name] is None) or
                            (p.grad is not None and captured[name] is not None and
                             torch.equal(p.grad.detach().cpu(), captured[name]))
                            for name, p in parameters.items() if name not in scope_names)
            if not unchanged:
                raise RuntimeError("Intervention altered a non-depth preclip gradient")
            branch_row = {"pre_update_state_sha256": pre_hash,
                "all_non_depth_preclip_gradients_exactly_unchanged": unchanged,
                "preclip_gradients": {name: tensor_stats(p.grad) for name, p in parameters.items()}}
            norm, used_coefficient = apply_clip(model, experiment.args.clip_norm,
                coefficient if branch == "remove_fixed_clip" else None)
            if branch == "normal_A":
                coefficient = used_coefficient
            branch_row.update(global_grad_preclip=norm, clip_coefficient=used_coefficient,
                postclip_gradients_sha256=digest({name: p.grad for name, p in parameters.items()}),
                depth_optimizer_before=momentum_report(optimizer, scope))
            optimizer.step()
            branch_row["depth_optimizer_after"] = momentum_report(optimizer, scope)
            branch_row["parameter_delta_from_checkpoint"] = parameter_deltas(model, initial["model_state_dict"], experiment.groups)
            after_hash = {"model": digest(model.state_dict()), "optimizer": digest(optimizer.state_dict()),
                          "rng": digest(dd.capture_rng_state()), "runtime": digest(capture_runtime(model)),
                          "loader": digest(experiment.stream.state_dict())}
            branch_row["post_update_state_sha256"] = after_hash
            references = {"before_step": before, "before_optimizer": pre_optimizer}
            if normal_parameters is not None:
                branch_row["parameter_delta_from_normal_A"] = parameter_deltas(model, normal_parameters, experiment.groups)
                references["normal_A"] = normal_probe
            after = probe(experiment, options.probe_modes)
            branch_row["after_update_probe"] = probe_report(after, references)
            first_per_split = {}
            for row in after:
                first_per_split.setdefault(row["split"], row)
            selected_maps[branch] = first_per_split
            if branch == "normal_A":
                normal_parameters = {name: p.detach().cpu().clone() for name, p in parameters.items()}
                normal_probe, normal_state_hash = after, after_hash
            if branch == "normal_B":
                branch_row["control_exact_state_match"] = after_hash == normal_state_hash
            report["branches"][branch] = branch_row
            dd.write_json(output / "counterfactual_report.json", driver.json_safe(report))
            print(f"[P4] {branch} complete; global norm={norm:.6g}, clip={used_coefficient:.6g}", flush=True)
        report["normal_repeat_exact_state_match"] = report["branches"]["normal_B"]["control_exact_state_match"]
        report["identical_pre_update_state_all_branches"] = len({json.dumps(row["pre_update_state_sha256"], sort_keys=True)
            for row in report["branches"].values()}) == 1
        fixed = report["branches"]["remove_fixed_clip"]["parameter_delta_from_normal_A"]["parameters"]
        report["fixed_clip_non_depth_updates_exactly_match_normal"] = all(
            fixed[name]["max_abs"] == 0.0 for name in fixed if name not in scope_names)
        torch.save({"before_step": before[:1], "after_forward_pre_optimizer": pre_optimizer[:1],
                    "branches_first_image_per_split": selected_maps}, output / "selected_probe_maps.pt")
        report["status"] = "completed" if all(report[key] for key in (
            "normal_repeat_exact_state_match", "identical_pre_update_state_all_branches",
            "fixed_clip_non_depth_updates_exactly_match_normal")) else "invalid_replay_control"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        restore_full_state(experiment, initial)
        unchanged_source = digest({key: initial[key] for key in (
            "model_state_dict", "optimizer_state_dict", "rng", "loader", "routes", "_replay_runtime")}) == source_digest
        report["source_state_unmodified"] = unchanged_source
        report["restored_checkpoint_model_exact"] = digest(model.state_dict()) == digest(initial["model_state_dict"])
        report["restored_checkpoint_optimizer_exact"] = digest(optimizer.state_dict()) == digest(initial["optimizer_state_dict"])
        report["restored_checkpoint_rng_exact"] = digest(dd.capture_rng_state()) == digest(initial["rng"])
        report["restored_loader_exact"] = digest(experiment.stream.state_dict()) == digest(initial["loader"])
        report["restored_runtime_exact"] = digest(capture_runtime(model)) == digest(initial["_replay_runtime"])
        report["restored_routes_exact"] = model.get_depth_grad_routes() == initial["routes"]
        report["cleared_parameter_gradients"] = all(parameter.grad is None for parameter in model.parameters())
        dd.write_json(output / "counterfactual_report.json", driver.json_safe(report))
    if report["status"] != "completed" or not all(report[key] for key in (
            "source_state_unmodified", "restored_checkpoint_model_exact", "restored_checkpoint_optimizer_exact",
            "restored_checkpoint_rng_exact", "restored_loader_exact", "restored_runtime_exact",
            "restored_routes_exact", "cleared_parameter_gradients")):
        raise RuntimeError("P4 replay control/restoration failed; inspect counterfactual_report.json")
    return report


def main(argv=None):
    options = parse_args(argv)
    output = Path(options.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {output}")
    import torch
    import numpy as np
    import depth_dynamics as dd
    import train_cva_depth_dynamics as driver
    if not torch.cuda.is_available():
        raise RuntimeError("Real P4 requires the project CUDA/custom-op environment")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    initial = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
    if initial.get("format_version") != driver.FORMAT_VERSION or "optimizer_state_dict" not in initial:
        raise ValueError("A complete dynamics checkpoint is required")
    driver.torch, driver.np, driver.dd = torch, np, dd
    driver.ORIGINAL_COMMAND = [sys.executable, *sys.argv]
    defaults, _ = driver.parse_args(["--init_checkpoint", options.init_checkpoint, "--output", str(output)])
    vars(defaults).update(initial["arguments"])
    defaults.init_checkpoint = options.init_checkpoint
    defaults.output, defaults.diagnostics_dir = str(output), str(output / "construction")
    defaults.resume_checkpoint = options.checkpoint
    defaults.mode, defaults.save_gradient_maps = "audit", False
    experiment = None
    try:
        experiment = driver.Experiment(defaults, config_cli(initial["cfg"]))
        run(experiment, initial, options)
    finally:
        if experiment is not None:
            experiment.trainer.close()


if __name__ == "__main__":
    main()
