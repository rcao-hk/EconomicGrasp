"""Locate repeat-backward noise on one unchanged real-batch graph (no updates)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import numpy as np
import depth_dynamics as dd
import train_cva_depth_dynamics as run


def main():
    args, remaining = run.parse_args()
    run.torch, run.np, run.dd = torch, np, dd
    run.ORIGINAL_COMMAND = [sys.executable, *sys.argv]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(args.deterministic_ops, warn_only=True)
    if args.deterministic_ops:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    experiment = run.Experiment(args, remaining)
    captured, handles = {}, []

    def hook(name):
        def capture(module, inputs, output):
            values = output if isinstance(output, (tuple, list)) else [output]
            for i, value in enumerate(values):
                if torch.is_tensor(value) and value.requires_grad:
                    captured[f"{name}:{i}"] = value
        return capture

    try:
        for name, module in experiment.model.named_modules():
            if name in ("proposal_head", "spatial_enhancer") or name.startswith((
                    "proposal_head.scratch.refinenet", "proposal_head.scratch.layer")) and name.count(".") == 2:
                handles.append(module.register_forward_hook(hook(name)))
        experiment.model.train()
        batch, indices = experiment.stream.next()
        loss, endpoints = experiment.forward(batch)
        targets = dict(captured)
        targets.update({name: p for name, p in experiment.model.named_parameters()
                        if name in ("proposal_head.projects.2.weight", "proposal_head.readout_projects.2.0.weight")})
        terms = dict(experiment.losses(endpoints), total=loss)
        result = {"batch_indices": indices, "optimizer_updates": 0, "same_forward_graph": True, "terms": {}}
        for term, objective in terms.items():
            first = torch.autograd.grad(objective, list(targets.values()), retain_graph=True, allow_unused=True)
            first = [None if g is None else g.detach().cpu().clone() for g in first]
            second = torch.autograd.grad(objective, list(targets.values()), retain_graph=True, allow_unused=True)
            rows = {}
            for name, a, b in zip(targets, first, second):
                if a is None or b is None:
                    rows[name] = {"none_first": a is None, "none_second": b is None}
                else:
                    b = b.detach().cpu()
                    rows[name] = {"exact": torch.equal(a, b), "max_abs_difference": float((a-b).abs().max()),
                                  "difference_l2": float((a-b).norm()), "gradient_l2": float(a.norm())}
            result["terms"][term] = rows
            print(term, {k: v for k, v in rows.items() if v.get("exact") is False}, flush=True)
        dd.write_json(experiment.diag / "same_graph_backward_repeat.json", result)
    finally:
        for handle in handles:
            handle.remove()
        experiment.trainer.close()


if __name__ == "__main__":
    main()
