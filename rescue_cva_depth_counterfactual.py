"""Bounded P4 trajectory from a full state: normal vs remove view->DPT/FiLM.

Each arm recomputes its own native forward and full gradient on every update.
The removal arm uses that SAME update's unmodified full-gradient clip factor;
it does not borrow a factor from the diverging normal trajectory. Adam history,
all forward routes, native matching and the original data stream are retained.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import replay_cva_depth_counterfactual as replay


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--branch", choices=("normal", "remove_view_fixed_clip"), required=True)
    parser.add_argument("--steps", type=int, choices=range(50, 201), default=100)
    return parser.parse_args()


def run(experiment, initial, options):
    import torch
    import depth_dynamics as dd
    import train_cva_depth_dynamics as driver

    model, optimizer = experiment.model, experiment.optimizer
    if not isinstance(optimizer, torch.optim.AdamW) or initial.get("scaler_state") is not None:
        raise ValueError("Rescue requires the FP32 AdamW dynamics state")
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    scope = [(name, p) for group, pairs in experiment.groups.items()
             if group == "depth_film" or group.startswith("depth_dpt.")
             for name, p in pairs if p.requires_grad]
    if not scope or any(p.dtype != torch.float32 for p in parameters.values()):
        raise ValueError("Expected FP32 trainable parameters and nonempty depth scope")
    initial["_replay_runtime"] = replay.capture_runtime(model)
    replay.restore_full_state(experiment, initial)
    experiment.contract["rescue"] = {
        "branch": options.branch, "steps": options.steps, "start_step": initial["step"],
        "checkpoint": str(Path(options.checkpoint).resolve()),
        "checkpoint_sha256": dd.sha256_file(options.checkpoint),
        "initial_state_sha256": {key: replay.digest(initial[key]) for key in
            ("model_state_dict", "optimizer_state_dict", "rng", "loader", "routes")},
        "scope_parameters": [name for name, _ in scope],
        "clip": "same-update full-gradient coefficient, before view subtraction",
        "limits": "native matching and all parameters evolve; non-depth gradients agree only at a common state; no momentum reset",
        "source_sha256": {str(Path(module.__file__).resolve()): dd.sha256_file(module.__file__)
                          for module in (sys.modules[__name__], replay, driver, dd)},
    }
    experiment.write_contract()
    experiment.probe()
    experiment.checkpoint("initial")
    model.train()
    for module in model.modules():
        if hasattr(module, "is_training"):
            module.is_training = True
    for update in range(1, options.steps + 1):
        batch, indices = experiment.stream.next()
        if experiment.args.lr_schedule == "source_cosine":
            experiment.trainer.adjust_learning_rate(min(experiment.initial_epoch + experiment.stream.epoch,
                                                        experiment.cfg.max_epoch))
        optimizer.zero_grad(set_to_none=True)
        loss, endpoints = experiment.forward(batch)
        target = experiment.losses(endpoints)["view"] * experiment.weights["view"]
        gradients = torch.autograd.grad(target, [p for _, p in scope], retain_graph=True, allow_unused=True)
        removed = {name: None if gradient is None else gradient.detach().cpu().clone()
                   for (name, _), gradient in zip(scope, gradients)}
        del gradients
        loss.backward()
        captured = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                    for name, p in parameters.items()}
        for values in (captured, removed):
            if any(value is not None and not bool(torch.isfinite(value).all()) for value in values.values()):
                raise FloatingPointError("Nonfinite total/target gradient")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf, error_if_nonfinite=True)
        coefficient = float((experiment.args.clip_norm / (norm + 1e-6)).clamp(max=1.0))
        replay.install_gradients(model, captured, removed if options.branch != "normal" else None)
        effective_norm, used = replay.apply_clip(model, experiment.args.clip_norm, coefficient)
        before = {name: p.detach().cpu().clone() for name, p in parameters.items()} if update % 10 == 0 else None
        optimizer.step()
        experiment.step += 1
        experiment.seen_images += len(indices)
        row = {"step": experiment.step, "relative_update": update, "branch": options.branch,
               "seen_images": experiment.seen_images, "batch_indices": indices,
               "batch_sha256": replay.digest(batch), "loader": experiment.stream.state_dict(),
               "loss": float(loss.detach()), "weighted_view_loss": float(target.detach()),
               "full_global_grad_preclip": float(norm), "effective_global_grad_preclip": effective_norm,
               "clip_coefficient": used, "target_connected_parameters": sum(v is not None for v in removed.values()),
               "coverage": experiment.scalar_metrics(endpoints)}
        if before is not None:
            row["parameter_update"] = replay.parameter_deltas(model, before, experiment.groups)["groups"]
        dd.append_jsonl(experiment.diag / "rescue_steps.jsonl", driver.json_safe(row))
        del loss, endpoints, target, captured, removed, before, batch
        if update % 10 == 0:
            print(f"[rescue] {options.branch} {update}/{options.steps} step={experiment.step}", flush=True)
        if update % 50 == 0 or update == options.steps:
            experiment.probe()
            experiment.checkpoint("final" if update == options.steps else "rolling")
    experiment.contract["completion"] = {"step": experiment.step, "status": "completed_rescue_budget",
                                          "seen_images": experiment.seen_images}
    experiment.write_contract()


def main():
    options = parse_args()
    output = Path(options.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output: {output}")
    import torch
    import numpy as np
    import depth_dynamics as dd
    import train_cva_depth_dynamics as driver
    if not torch.cuda.is_available():
        raise RuntimeError("Rescue requires the project CUDA environment")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    initial = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
    if initial.get("format_version") != driver.FORMAT_VERSION or "optimizer_state_dict" not in initial:
        raise ValueError("Expected a complete dynamics checkpoint")
    driver.torch, driver.np, driver.dd = torch, np, dd
    driver.ORIGINAL_COMMAND = [sys.executable, *sys.argv]
    defaults, _ = driver.parse_args(["--init_checkpoint", options.init_checkpoint, "--output", str(output)])
    vars(defaults).update(initial["arguments"])
    defaults.init_checkpoint = options.init_checkpoint
    defaults.output, defaults.diagnostics_dir = str(output), str(output / "diagnostics")
    defaults.resume_checkpoint = options.checkpoint
    defaults.max_steps = initial["step"] + options.steps
    defaults.rescue_branch, defaults.rescue_steps = options.branch, options.steps
    defaults.mode, defaults.save_gradient_maps = "train", False
    experiment = None
    try:
        experiment = driver.Experiment(defaults, replay.config_cli(initial["cfg"]))
        run(experiment, initial, options)
    except BaseException as error:
        if experiment is not None:
            dd.write_json(experiment.diag / "rescue_failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if experiment is not None:
            experiment.trainer.close()


if __name__ == "__main__":
    main()
