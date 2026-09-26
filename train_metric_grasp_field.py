#!/usr/bin/env python3
"""Online RGB -> DAV2 metric grasp field training. No P0/P1 mining cache.

The DAV2 encoder and original relative decoder are frozen. Metric DPT and ray
profile head train on geometry losses; grasp losses cannot enter either branch.
"""
import argparse
from datetime import timedelta
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from metric_grasp_field_core import MetricFieldConfig
from metric_field_runtime import (
    VERSION, BASE_MAIN_SHA, BASE_CONFIG, DEFAULT_WORK, atomic_json, atomic_torch,
    code_fingerprint, configure_base, construct_model, dataset_schedule, digest,
    make_dataset, move_batch, restore_rng, rng_state, seed_all, sha256_file, worker_init,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default="/data/robotarm/dataset/graspnet")
    p.add_argument("--output-root", default=DEFAULT_WORK + "/train")
    p.add_argument("--init-checkpoint", default="", help="Trusted main RGB-CDF checkpoint; empty starts new metric/task heads")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--encoder", choices=("vits", "vitb", "vitl"), default="vitb")
    p.add_argument("--pose-mode", choices=("none", "global_film", "ray_gravity_film"), default="global_film")
    p.add_argument("--seed-mode", choices=("image_fps", "point_fps"), default="image_fps")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1, help="Per GPU")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--geometry-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--sample-fraction", type=float, default=.1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--eval-workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=2117)
    p.add_argument("--m-point", type=int, default=1024)
    p.add_argument("--group-chunk", type=int, default=256)
    p.add_argument("--action-chunk", type=int, default=256)
    p.add_argument("--field-bins", type=int, default=160)
    p.add_argument("--field-hidden", type=int, default=64)
    p.add_argument("--field-stride", type=int, default=4)
    p.add_argument("--evidence-mode", choices=("hard", "fixed", "learned"), default="learned")
    p.add_argument("--surface-epsilon", type=float, default=.005)
    p.add_argument("--prior-sigma", type=float, default=.03)
    p.add_argument("--fixed-sigma", type=float, default=.02)
    p.add_argument("--profile-weight", type=float, default=1.)
    p.add_argument("--profile-mean-weight", type=float, default=10.)
    p.add_argument("--base-cdf-weight", type=float, default=.25)
    p.add_argument("--max-train-frames", type=int, default=0)
    p.add_argument("--max-val-frames", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0, help="Per epoch smoke cap; marks run partial")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--amp", action="store_true", help="Opt-in CUDA mixed precision; default FP32")
    p.add_argument("--no-chunk-checkpoint", action="store_true")
    return p


def grad_norm(grads):
    return math.sqrt(sum(float(g.detach().float().square().sum()) for g in grads if g is not None))


def gradient_contract(model, batch, loss_fn, loss_kwargs):
    """Real forward graph check, including latent bypasses, not just one detach()."""
    model.train()
    ep = model(batch)
    _, stats = loss_fn(ep, model.config, **loss_kwargs)
    geometry_params = model.geometry_parameters()
    readout_params = [p for p in model.readout.parameters() if p.requires_grad]
    task_geom = torch.autograd.grad(stats["task_loss"], geometry_params,
                                    retain_graph=True, allow_unused=True)
    task_numeric = torch.autograd.grad(stats["task_loss"],
                                       [ep["depth_map_pred"], ep["mgf_profile_logits"]],
                                       retain_graph=True, allow_unused=True)
    geom_grad = torch.autograd.grad(stats["geometry_loss"], geometry_params,
                                    retain_graph=True, allow_unused=True)
    reader_grad = torch.autograd.grad(stats["task_loss"], readout_params,
                                      retain_graph=False, allow_unused=True)
    report = {
        "grasp_to_geometry_parameters_norm": grad_norm(task_geom),
        "grasp_to_predicted_depth_norm": grad_norm(task_numeric[:1]),
        "grasp_to_depth_profile_norm": grad_norm(task_numeric[1:]),
        "geometry_to_geometry_parameters_norm": grad_norm(geom_grad),
        "grasp_to_field_reader_norm": grad_norm(reader_grad),
        "cdf_valid_candidates": int(ep["batch_grasp_cdf_valid_mask"].sum()),
        "frozen_encoder": all(not p.requires_grad for p in model.base.depth_net.depthnet.pretrained.parameters()),
        "frozen_relative_decoder": all(not p.requires_grad for p in model.relative_decoder.parameters()),
        "metric_parameters_trainable": len(geometry_params),
    }
    if any(report[k] != 0. for k in ("grasp_to_geometry_parameters_norm",
                                     "grasp_to_predicted_depth_norm", "grasp_to_depth_profile_norm")):
        raise RuntimeError(f"Detach contract violated: {report}")
    if not report["frozen_encoder"] or not report["frozen_relative_decoder"]:
        raise RuntimeError("Foundation parameters unexpectedly trainable")
    if not math.isfinite(report["geometry_to_geometry_parameters_norm"]) or report["geometry_to_geometry_parameters_norm"] <= 0:
        raise RuntimeError("No finite geometry supervision gradient")
    if report["cdf_valid_candidates"] > 0 and report["grasp_to_field_reader_norm"] <= 0:
        raise RuntimeError("Field reader has valid labels but no gradient")
    return report


def scalar_stats(stats):
    out = {k: float(v.detach()) for k, v in stats.items()}
    if not all(math.isfinite(v) for v in out.values()):
        raise FloatingPointError(f"Nonfinite metrics: {out}")
    return out


def reduce_sums(sums, count, device, world):
    keys = sorted(sums)
    pack = torch.tensor([sums[k] for k in keys] + [count], device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(pack)
    return {k: float(pack[i] / pack[-1].clamp_min(1)) for i, k in enumerate(keys)}


@torch.no_grad()
def validate(model, loader, device, loss_fn, kwargs):
    model.eval()
    sums, count, matched = {}, 0, 0
    for raw in loader:
        batch = move_batch(raw, device)
        ep = model(batch)
        _, stats = loss_fn(ep, model.config, **kwargs)
        values = scalar_stats(stats)
        b = len(batch["img"])
        for k, v in values.items():
            sums[k] = sums.get(k, 0.) + b*v
        matched += int(ep["batch_grasp_cdf_valid_mask"].sum())
        count += b
    if not count or not matched:
        raise RuntimeError("Validation has no frames/valid CDF labels; cannot select checkpoint")
    return {**{k: v/count for k, v in sums.items()}, "frames": count,
            "valid_cdf_candidates": matched}


def main():
    args = parser().parse_args()
    if min(args.epochs, args.batch_size, args.log_every, args.m_point) < 1:
        raise ValueError("Epoch/batch/log/query counts must be positive")
    if min(args.max_steps, args.max_train_frames, args.max_val_frames, args.workers, args.eval_workers) < 0:
        raise ValueError("Frame/step/worker counts cannot be negative")
    if min(args.lr, args.geometry_lr) <= 0 or min(args.weight_decay, args.profile_weight,
                                                args.profile_mean_weight, args.base_cdf_weight) < 0:
        raise ValueError("Invalid loss/optimizer settings")
    if args.profile_weight == 0 or args.profile_mean_weight == 0:
        raise ValueError("The metric profile needs geometry supervision; keep both profile weights positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Full main-based training requires the CUDA/GraspNet environment; CPU tests are separate")
    world, rank, local = int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0")), int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
    seed_all(args.seed + rank)
    config = MetricFieldConfig(
        bins=args.field_bins, hidden=args.field_hidden, field_stride=args.field_stride,
        action_chunk=args.action_chunk, checkpoint_chunks=not args.no_chunk_checkpoint,
        evidence_mode=args.evidence_mode, surface_epsilon=args.surface_epsilon,
        prior_sigma=args.prior_sigma, fixed_sigma=args.fixed_sigma,
    )
    base_cfg = {**BASE_CONFIG, "m_point": args.m_point, "kview_group_chunk": args.group_chunk}
    configure_base(base_cfg, config, args.pose_mode)
    from dataset.graspnet_dataset import collate_fn
    from models.economicgrasp_metric_field import metric_field_loss
    dataset, train_set, train_idx = make_dataset(args.dataset_root, "train", args.sample_fraction,
                                                labels=True, max_frames=args.max_train_frames)
    val_full, val_set, val_idx = make_dataset(args.dataset_root, "test_seen", args.sample_fraction,
                                             labels=True, max_frames=args.max_val_frames)
    if not len(train_set) or not len(val_set):
        raise RuntimeError("Empty data schedule")
    official = Path("checkpoints") / f"depth_anything_v2_{args.encoder}.pth"
    loss_kwargs = {k: getattr(args, k) for k in ("profile_weight", "profile_mean_weight", "base_cdf_weight")}
    sampling = {"train": dataset_schedule(dataset, train_idx), "test_seen": dataset_schedule(val_full, val_idx)}
    protocol = {
        "version": VERSION, "base_main_sha": BASE_MAIN_SHA, "code_sha256": code_fingerprint(),
        "encoder": args.encoder, "pose_mode": args.pose_mode, "seed_mode": args.seed_mode,
        "field": asdict(config), "base_config": base_cfg, "sample_fraction": args.sample_fraction,
        "sampling_sha256": digest(sampling), "train_frames": len(train_idx), "val_frames": len(val_idx),
        "init_checkpoint": str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else "",
        "init_sha256": sha256_file(args.init_checkpoint) if args.init_checkpoint else "",
        "dav2_sha256": sha256_file(official), "loss_weights": loss_kwargs,
        "optimizer": {"task_lr": args.lr, "geometry_lr": args.geometry_lr,
                      "weight_decay": args.weight_decay, "batch_per_gpu": args.batch_size,
                      "world_size": world, "effective_batch": args.batch_size * world,
                      "amp": args.amp},
        "partial_run": bool(args.max_train_frames or args.max_val_frames or args.max_steps),
        "max_steps": args.max_steps, "seed": args.seed,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "torch_version": str(torch.__version__),
        "ddp_training_padding_frames": (math.ceil(len(train_idx)/world)*world-len(train_idx)),
        "checkpoint_selection": "lowest Seen field-CDF BCE, not a no-op policy metric",
        "depth_contract": "predicted depth/profile/metric latent detached from all grasp losses; geometry supervision remains live",
        "label_contract": "main online <=5mm NN canonical CDF/width annotation matching, NOT arbitrary-action DexNet labels",
    }
    signature = digest(protocol)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    # Single writer/rank, advisory lock released automatically even after a crash.
    lock_file = None
    if rank == 0:
        import fcntl
        lock_file = open(out/".train.lock", "a")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    latest = out/"checkpoint_latest.pt"
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError(f"--resume requested but missing {latest}")
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        if ck["signature"] != signature:
            raise RuntimeError("Resume protocol/code/checkpoint/sampling changed; use a new output root")
    else:
        ck = None
        if (out/"protocol.json").exists() or latest.exists():
            raise FileExistsError(f"Output already used: {out}; --resume or a NEW output root required")
    model = construct_model(protocol, initialize=not args.resume, device=device)
    geom_params = model.geometry_parameters()
    geom_ids = {id(p) for p in geom_params}
    task_params = [p for p in model.parameters() if p.requires_grad and id(p) not in geom_ids]
    optimizer = torch.optim.AdamW([
        {"params": task_params, "lr": args.lr, "name": "grasp"},
        {"params": geom_params, "lr": args.geometry_lr, "name": "geometry"}], weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    start, best, history = 0, float("inf"), []
    if ck:
        model.load_state_dict(ck["model"], strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start, best, history = ck["epoch"]+1, ck["best_cdf"], ck["history"]
    train_sampler = DistributedSampler(train_set, world, rank, shuffle=True, seed=args.seed) if world > 1 else None
    generator = torch.Generator().manual_seed(args.seed + rank)
    loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                        shuffle=train_sampler is None, num_workers=args.workers, collate_fn=collate_fn,
                        worker_init_fn=worker_init, generator=generator, pin_memory=False,
                        persistent_workers=False, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.eval_workers, collate_fn=collate_fn,
                            worker_init_fn=worker_init, pin_memory=False) if rank == 0 else None
    if rank == 0:
        # Restore RNG + buffers after audit so diagnostics do not change training.
        rng_before = rng_state()
        buffers_before = {k: v.clone() for k, v in model.named_buffers()}
        first = collate_fn([train_set[0]])
        report = gradient_contract(model, move_batch(first, device), metric_field_loss, loss_kwargs)
        atomic_json(out/"gradient_contract.json", report)
        restore_rng(rng_before)
        for k, v in model.named_buffers():
            v.copy_(buffers_before[k])
        del first, buffers_before
        if not args.resume:
            atomic_json(out/"protocol.json", protocol)
            atomic_json(out/"sampling.json", sampling)
        print("[MGF GRADIENT] " + json.dumps(report), flush=True)
    if world > 1:
        dist.barrier()
    if ck:
        restore_rng(ck["rng_by_rank"][rank])
        generator.set_state(ck["loader_rng_by_rank"][rank])
        del ck
    train_model = DDP(model, device_ids=[local], find_unused_parameters=True,
                      broadcast_buffers=False) if world > 1 else model
    total_batches = min(len(loader), args.max_steps) if args.max_steps else len(loader)
    for epoch in range(start, args.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        sums, count = {}, 0
        t0 = time.monotonic()
        for step, raw in enumerate(loader):
            if step >= total_batches:
                break
            batch = move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=args.amp):
                ep = train_model(batch)
                loss, stats = metric_field_loss(ep, config, **loss_kwargs)
            values = scalar_stats(stats)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                task_params + geom_params, 5., error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            b = len(batch["img"])
            for k, v in values.items():
                sums[k] = sums.get(k, 0.) + b*v
            count += b
            if rank == 0 and (step+1) % args.log_every == 0:
                print(f"[MGF TRAIN] epoch={epoch} step={step+1}/{total_batches} "
                      f"loss={values['loss']:.5f} cdf={values['cdf']:.5f} "
                      f"depth={values['depth_l1']:.5f}", flush=True)
            del ep, batch, loss, stats
        train_stats = reduce_sums(sums, count, device, world)
        # Rank-0 validation uses unwrapped model, avoiding DDP collective padding.
        val_stats = validate(model, val_loader, device, metric_field_loss, loss_kwargs) if rank == 0 else None
        if world > 1:
            dist.barrier()
        state = rng_state()
        loader_state = generator.get_state()
        states, loader_states = [None]*world, [None]*world
        if world > 1:
            dist.all_gather_object(states, state)
            dist.all_gather_object(loader_states, loader_state)
        else:
            states[0], loader_states[0] = state, loader_state
        if rank == 0:
            row = {"epoch": epoch, "train": train_stats, "validation": val_stats,
                   "seconds": time.monotonic()-t0}
            history.append(row)
            improved = val_stats["cdf"] < best
            best = min(best, val_stats["cdf"])
            payload = {"version": VERSION, "signature": signature, "protocol": protocol,
                       "epoch": epoch, "best_cdf": best, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                       "history": history, "rng_by_rank": states, "loader_rng_by_rank": loader_states}
            atomic_torch(latest, payload)
            if improved:
                atomic_torch(out/"checkpoint_best.pt", payload)
                atomic_json(out/"best.json", row)
            atomic_json(out/"metrics.json", history)
            print("[MGF EPOCH] " + json.dumps(row), flush=True)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()
    if lock_file is not None:
        lock_file.close()


if __name__ == "__main__":
    main()
