#!/usr/bin/env python3
"""Train Rep-B fixed-action hypothesis-conditioned representations.

Formal first-round variants:
  B0: relational RGB hypotheses, no predicted-depth prior.
  B1: B0 + soft predicted-depth prior, nominal-depth training.
  B2: B0 + soft predicted-depth prior, depth-error augmentation training.

All use identical fixed Rep-P0 actions and exact-action labels from Rep-A cache.
No candidate generation, CAD evaluator, or depth predictor is trained here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from rep_a_common import (
    check_runtime_sources,
    digest,
    exclusive_run,
    file_sha,
    list_frames,
    load_torch,
    metrics,
    perturb_depth,
    read_frame,
    save_json,
    save_torch,
    seed_for,
    training_case,
    tune_margin,
)
from rep_b_common import cdf_bce_loss, pairwise_ranking_loss, rep_b_tensors
from rep_b_model import REP_B_VARIANTS, RepBModel


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--variant", choices=REP_B_VARIANTS, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-train-frames", type=int, default=0)
    p.add_argument("--max-val-frames", type=int, default=0)
    p.add_argument("--dim", type=int, default=0, help="0 = Rep-A reader out_dim")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--prior-sigma-mm", type=float, default=30.0)
    p.add_argument("--pairwise-weight", type=float, default=0.0)
    p.add_argument("--pairwise-temperature", type=float, default=0.1)
    p.add_argument("--pairwise-min-gap", type=float, default=1e-4)
    p.add_argument("--max-bias-mm", type=float, default=20.0)
    p.add_argument("--max-scale", type=float, default=0.03)
    p.add_argument("--smooth-mm", type=float, default=10.0)
    p.add_argument("--nominal-prob", type=float, default=0.25)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--progress-every", type=int, default=100)
    return p


@torch.no_grad()
def validation(model, paths, contract, device):
    model.eval()
    predictions, frames = [], []
    for path in paths:
        d = read_frame(path, contract)
        t = rep_b_tensors(d, device)
        prob = model(t).sigmoid().cpu().numpy()
        predictions.append(prob)
        frames.append({k: d[k] for k in ("valid", "utility", "friction", "zero_index")})
        del d, t, prob
    return tune_margin(predictions, frames)


def main():
    args = parser().parse_args()
    sys.argv = [sys.argv[0]]
    if args.epochs < 1 or args.grad_accum_steps < 1:
        raise ValueError("epochs and grad-accum-steps must be positive")
    if args.lr <= 0 or args.prior_sigma_mm <= 0:
        raise ValueError("lr/prior-sigma-mm must be positive")
    if args.pairwise_weight < 0:
        raise ValueError("pairwise-weight must be nonnegative")
    if min(args.max_bias_mm, args.max_scale, args.smooth_mm) < 0:
        raise ValueError("augmentation amplitudes must be nonnegative")
    if not 0 <= args.nominal_prob <= 1:
        raise ValueError("nominal-prob must be in [0,1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly use --device cpu for a CPU smoke test")

    manifest = check_runtime_sources(args.cache_root)
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    init_path = Path(args.cache_root) / "reader_init.pt"
    init = load_torch(init_path)
    contract = init["contract"]
    channels = int(init["model_config"]["channels"])
    dim = int(args.dim or init["model_config"]["out_dim"])

    train_paths = list_frames(args.cache_root, "train", args.max_train_frames)
    val_paths = list_frames(args.cache_root, "test_seen", args.max_val_frames)

    model_spec = {
        "variant": args.variant,
        "channels": channels,
        "dim": dim,
        "heads": args.heads,
        "layers": args.layers,
        "dropout": args.dropout,
        "prior_sigma_mm": args.prior_sigma_mm,
    }
    config = {
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_accum_steps": args.grad_accum_steps,
        "grad_clip": args.grad_clip,
        "pairwise_weight": args.pairwise_weight,
        "pairwise_temperature": args.pairwise_temperature,
        "pairwise_min_gap": args.pairwise_min_gap,
        "max_bias_mm": args.max_bias_mm,
        "max_scale": args.max_scale,
        "smooth_mm": args.smooth_mm,
        "nominal_prob": args.nominal_prob,
        "model_spec": model_spec,
    }
    signature = digest({
        "config": config,
        "contract": contract,
        "manifest": manifest,
        "rep_b_model_sha": file_sha(Path(__file__).resolve().parent / "rep_b_model.py"),
        "rep_a_model_sha": file_sha(Path(__file__).resolve().parent / "rep_a_model.py"),
        "frames": [
            (str(p.relative_to(args.cache_root)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in train_paths + val_paths
        ],
    })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with exclusive_run(out / ".train.lock"):
        # Same seed + common-module allocation order gives paired initialization
        # for B0/B1/B2 shared parameters.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = RepBModel(**model_spec).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        latest = out / "checkpoint_latest.pt"
        start_epoch = 0
        history = []
        best_key = (-float("inf"), -float("inf"), -float("inf"))

        if latest.exists():
            if not args.resume:
                raise FileExistsError(f"{latest}; use --resume or a new output-dir")
            ck = load_torch(latest)
            if ck["signature"] != signature:
                raise RuntimeError("Rep-B training/data contract changed; cannot resume")
            model.load_state_dict(ck["model"], strict=True)
            optimizer.load_state_dict(ck["optimizer"])
            start_epoch = int(ck["epoch"]) + 1
            history = ck["history"]
            best_key = tuple(ck["best_key"])
            torch.set_rng_state(ck["torch_rng"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ck["cuda_rng"])
            del ck

        save_json(out / "protocol.json", {
            "experiment": "Rep-B hypothesis-conditioned image representation",
            "variant": args.variant,
            "signature": signature,
            "cache_contract": contract,
            "config": config,
            "epochs_requested": args.epochs,
            "train_frames": len(train_paths),
            "val_frames": len(val_paths),
            "sample_interval": "inherited from Rep-A/Rep-P0 cache (formal cache=0.1)",
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_evidence": "cached pre-enhancer image feature and predicted metric depth",
            "fixed": "K-ray physical actions, exact-action labels, image backbone, depth estimator",
            "depth_use": (
                "none" if not model.use_prior
                else "soft per-candidate prior only; never RGB projection/masking"
            ),
            "augmentation": bool(model.augment_depth),
            "validation": "nominal predicted depth only; checkpoint/margin selected on test_seen",
            "pairwise_loss_formal_default": args.pairwise_weight == 0,
            "not_official_AP": True,
        })

        for epoch in range(start_epoch, args.epochs):
            model.train()
            order = list(train_paths)
            random.Random(seed_for(args.seed, "rep_b_order", epoch)).shuffle(order)
            total_loss = total_cdf = total_pair = 0.0
            seen = updates = 0
            case_counts = {}

            for offset in range(0, len(order), args.grad_accum_steps):
                chunk = order[offset:offset + args.grad_accum_steps]
                optimizer.zero_grad(set_to_none=True)
                for path in chunk:
                    d = read_frame(path, contract)
                    t = rep_b_tensors(d, device)

                    if model.augment_depth:
                        s = seed_for(
                            args.seed, "rep_b_train", epoch,
                            int(d["scene_id"]), int(d["anno_id"])
                        )
                        case = training_case(
                            s,
                            args.max_bias_mm,
                            args.max_scale,
                            args.smooth_mm,
                            args.nominal_prob,
                        )
                        depth, _ = perturb_depth(t["depth"], case, s)
                    else:
                        case = "nominal"
                        depth = t["depth"]

                    logits = model(t, depth=depth)
                    cdf = cdf_bce_loss(logits, d["friction"], t["valid"])
                    pair = pairwise_ranking_loss(
                        logits,
                        d["utility"],
                        t["valid"],
                        min_gap=args.pairwise_min_gap,
                        temperature=args.pairwise_temperature,
                    )
                    loss = cdf + args.pairwise_weight * pair
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite Rep-B loss: {path} / {case}")

                    (loss / len(chunk)).backward()
                    total_loss += float(loss.detach())
                    total_cdf += float(cdf.detach())
                    total_pair += float(pair.detach())
                    seen += 1
                    kind = case.split(":")[0]
                    case_counts[kind] = case_counts.get(kind, 0) + 1
                    del d, t, depth, logits, cdf, pair, loss

                if epoch == start_epoch and updates == 0:
                    grad = {}
                    for name, part in model.named_children():
                        params = [p for p in part.parameters() if p.requires_grad]
                        if not params:
                            continue
                        value = sum(
                            float(p.grad.detach().norm())
                            for p in params if p.grad is not None
                        )
                        grad[name] = value
                    missing = [k for k, v in grad.items() if not np.isfinite(v) or v <= 0]
                    if missing:
                        raise RuntimeError(f"Missing/non-finite Rep-B component gradients: {grad}")
                    save_json(out / "gradient_check.json", grad)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
                updates += 1

                if seen % max(args.progress_every, 1) < len(chunk):
                    print(
                        f"[REP-B {args.variant}] epoch={epoch} "
                        f"frames={seen}/{len(order)} "
                        f"loss={total_loss/seen:.5f} cdf={total_cdf/seen:.5f} "
                        f"pair={total_pair/seen:.5f}",
                        flush=True,
                    )

            best, sweep = validation(model, val_paths, contract, device)
            key = (
                best["selected_utility"],
                -best["harm08"],
                -best["move_rate"],
            )
            improved = key > best_key
            if improved:
                best_key = key

            row = {
                "epoch": epoch,
                "train_loss": total_loss / seen,
                "train_cdf_loss": total_cdf / seen,
                "train_pairwise_loss": total_pair / seen,
                "updates": updates,
                "augmentation_counts": case_counts,
                "val": best,
            }
            history.append(row)
            payload = {
                "experiment": "Rep-B",
                "variant": args.variant,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "signature": signature,
                "contract": contract,
                "model_spec": model_spec,
                "margin": float(best["margin"]),
                "history": history,
                "best_key": best_key,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            }
            if improved:
                save_torch(out / "checkpoint_best.pt", payload)
                save_json(out / "best.json", row)
                save_json(out / "margin_sweep_best.json", sweep)
            save_torch(latest, payload)
            save_json(out / "metrics.json", history)
            print(json.dumps(row, sort_keys=True), flush=True)

        print(f"[REP-B {args.variant}] finished: {out/'checkpoint_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
