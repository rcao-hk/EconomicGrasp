"""Fine-tune RGB-only CVA/CDF with GT-anchor relative-depth supervision.

Three matched controls: none (original metric L1), foreground pairs, anchor
pairs. Geometry detach and frozen DINO are retained in every arm. Use torchrun
for DDP. Checkpoints remain loadable by the existing Stage-1 inference script.
"""

import argparse
import copy
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from cva_depth_geometry import (depth_metrics, matched_depth_pairs, metric_depth_loss,
                                pair_metrics, relative_depth_loss)
from cva_depth_experiment import (GRASP_WEIGHT_DEFAULTS, add_common_arguments, append_jsonl, checkpoint_metadata,
                                  grasp_objective, load_model, make_dataset, move_batch,
                                  pair_config, seed_all, seed_worker, validate_output_dir, write_json)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--variant", choices=("none", "foreground", "anchor"), default="anchor")
    parser.add_argument("--train_scope", choices=("joint", "depth_only"), default="joint")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2, help="Per GPU.")
    parser.add_argument("--depth_lr", type=float, default=1e-5)
    parser.add_argument("--grasp_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--metric_depth_weight", type=float, default=10.0)
    parser.add_argument("--relative_weight", type=float, default=10.0)
    parser.add_argument("--relative_warmup_epochs", type=int, default=1)
    parser.add_argument("--depth_normalization", choices=("full_image", "valid_pixels"), default="full_image")
    parser.add_argument("--clip_mode", choices=("global", "separate"), default="global")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=0, help="Evenly select a smaller training subset; 0 means all.")
    parser.add_argument("--max_steps_per_epoch", type=int, default=0, help="Smoke-test cap; 0 means all batches.")
    parser.add_argument("--eval_split", choices=("train", "test_seen", "test_similar", "test_novel"), default="train")
    parser.add_argument("--eval_scene_ids", default=None,
                        help="Validation scenes, separate from --scene_ids. Default: 90..99 for train split, all for other splits.")
    parser.add_argument("--eval_max_frames", type=int, default=64)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.eval_scene_ids is None:
        args.eval_scene_ids = ",".join(map(str, range(90, 100))) if args.eval_split == "train" else ""
    if args.eval_split == "train" and not args.eval_scene_ids.strip():
        parser.error("Training-split validation needs explicit held-out --eval_scene_ids.")
    if min(args.epochs, args.batch_size, args.log_every) < 1:
        parser.error("epochs, batch_size and log_every must be positive.")
    if min(args.depth_lr, args.grasp_lr, args.metric_depth_weight, args.max_grad_norm) <= 0:
        parser.error("Learning rates, metric weight and max_grad_norm must be positive.")
    if min(args.relative_weight, args.relative_warmup_epochs, args.max_steps_per_epoch, args.weight_decay,
           args.num_workers, args.max_frames, args.eval_max_frames) < 0:
        parser.error("Weights, warmup and step cap must be nonnegative.")
    return args


class DepthTrainingModule(torch.nn.Module):
    def __init__(self, model, args, cfgs):
        super().__init__()
        self.model, self.args, self.cfgs = model, args, cfgs
        self.pair_cfg = pair_config(args)
        if args.train_scope == "depth_only":
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for name, parameter in model.depth_net.named_parameters():
                if not name.startswith("depthnet.pretrained."):
                    parameter.requires_grad_(True)

    def set_training(self, training):
        self.train(training)
        active_grasp_training = training and self.args.train_scope == "joint"
        self.model.is_training = active_grasp_training
        self.model.view.is_training = active_grasp_training
        if self.args.train_scope == "depth_only":
            self.model.eval()
            self.model.depth_net.train(training)

    def forward(self, batch, pair_seed, relative_multiplier=1.0):
        if self.args.train_scope == "joint":
            ep = dict(batch)
            ep["cva_force_process_grasp_labels"] = True
            ep["cva_compute_diagnostics"] = False
            ep["rgb_geometry_compute_diagnostics"] = False
            ep = self.model(ep)
            prediction = ep["depth_map_pred"]
            task, _ = grasp_objective(ep, self.cfgs)
        else:
            prediction = self.predict_depth(batch)
            task = prediction.new_zeros(())
        pairs = matched_depth_pairs(batch, self.pair_cfg, pair_seed)
        absolute = metric_depth_loss(prediction, batch["gt_depth_m"], self.pair_cfg, self.args.depth_normalization)
        relative = (relative_depth_loss(prediction, batch["gt_depth_m"], pairs, self.args.variant, self.pair_cfg)
                    if self.args.variant != "none" else prediction.sum() * 0.0)
        relative_weight = self.args.relative_weight * relative_multiplier if self.args.variant != "none" else 0.0
        loss = task + self.args.metric_depth_weight * absolute + relative_weight * relative
        return {"loss": loss, "task": task, "absolute": absolute, "relative": relative,
                "prediction": prediction.detach(), "pairs": pairs}

    def predict_depth(self, batch):
        return self.model.depth_net(
            batch["img"], camera_pose_vec=batch.get(self.model.camera_pose_key),
            camera_gravity_vec=batch.get(self.model.camera_gravity_key), camera_K=batch["K"])[0]


def parameter_groups(module):
    depth = [p for p in module.model.depth_net.parameters() if p.requires_grad]
    ids = {id(p) for p in depth}
    grasp = [p for p in module.parameters() if p.requires_grad and id(p) not in ids]
    return depth, grasp


def gradient_norm(parameters):
    values = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    return torch.stack(values).norm() if values else torch.tensor(0.0, device=parameters[0].device if parameters else "cpu")


def checked_resume(checkpoint, args, train_indices, eval_indices, world_size):
    if checkpoint.get("depth_geometry_contract_version") != 1:
        raise ValueError("--resume requires a checkpoint created by this trainer. Omit --resume to initialize from Stage 1/2.")
    saved = checkpoint["depth_geometry_args"]
    keys = ("variant", "train_scope", "epochs", "batch_size", "depth_lr", "grasp_lr", "weight_decay",
            "metric_depth_weight", "relative_weight", "relative_warmup_epochs", "depth_normalization",
            "clip_mode", "max_grad_norm", "seed", "anchors_per_image", "pairs_per_anchor",
            "pair_radius_min_m", "pair_radius_max_m", "visibility_tolerance_m", "control_depth_tolerance_m",
            "relative_huber_beta_m", "max_steps_per_epoch", "cdf_label_folder", "graspness_mode",
            "eval_split", "eval_scene_ids", "dataset_root", "num_workers")
    keys += tuple(f"{name}_loss_weight" for name in GRASP_WEIGHT_DEFAULTS)
    differences = [key for key in keys if saved.get(key) != getattr(args, key)]
    if (differences or checkpoint.get("world_size") != world_size
            or checkpoint.get("train_indices") != train_indices or checkpoint.get("eval_indices") != eval_indices):
        raise ValueError(f"Resume experiment contract changed: {differences}; also check world size and selected frames.")


def save_checkpoint(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def main(argv=None):
    args = parse_args(argv)
    cfg = pair_config(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Training needs the repository's CUDA environment; CPU tensor tests are available separately.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    try:
        seed_all(args.seed + rank)
        model, checkpoint, cfgs = load_model(args, device, training=True)
        module = DepthTrainingModule(model, args, cfgs)
        depth_params, grasp_params = parameter_groups(module)
        groups = [{"params": depth_params, "lr": args.depth_lr}]
        if grasp_params:
            groups.append({"params": grasp_params, "lr": args.grasp_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        from dataset.graspnet_dataset import collate_fn
        eval_args = copy.copy(args)
        eval_args.scene_ids = args.eval_scene_ids
        held_out = {int(s) for s in args.eval_scene_ids.split(",") if s.strip()} if args.eval_split == "train" else set()
        train_data, train_indices = make_dataset(args, "train", args.max_frames, excluded_scenes=held_out)
        eval_data, eval_indices = make_dataset(eval_args, args.eval_split, args.eval_max_frames)
        start_epoch, best = 0, float("inf")
        if args.resume:
            checked_resume(checkpoint, args, train_indices, eval_indices, world_size)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint["epoch"])
            best = float(checkpoint["best_validation_geometry"])
            if start_epoch >= args.epochs:
                raise ValueError("Checkpoint has already reached --epochs.")
        if rank == 0:
            validate_output_dir(args.output_dir, resume=args.resume)
            write_json(Path(args.output_dir) / "contract.json", {**checkpoint_metadata(model, args),
                       "pair_config": cfg.to_dict(), "world_size": world_size,
                       "train_indices": train_indices, "eval_indices": eval_indices,
                       "geometry_detach": True, "dino_frozen": True, "dexnet_calls": 0,
                       "validation_selection": "fixed GT anchor geometry metrics; not test AP",
                       "resume_granularity": "completed epoch; sampler and pair seeds reset by epoch"})
        if world_size > 1:
            dist.barrier()
        output = Path(args.output_dir)
        sampler = DistributedSampler(train_data, shuffle=True, seed=args.seed) if world_size > 1 else None
        generator = torch.Generator()
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                                  num_workers=args.num_workers, collate_fn=collate_fn, worker_init_fn=seed_worker,
                                  generator=generator, drop_last=False, pin_memory=True)
        # Validation is sharded without padding/duplicates and runs the unwrapped
        # module. Uneven shard lengths therefore never deadlock DDP forwards.
        eval_loader = DataLoader(Subset(eval_data, list(range(rank, len(eval_data), world_size))),
                                 batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 collate_fn=collate_fn, worker_init_fn=seed_worker, pin_memory=True)
        network = DistributedDataParallel(module, device_ids=[local_rank], find_unused_parameters=True,
                                          broadcast_buffers=False) if world_size > 1 else module
        del checkpoint
        for epoch in range(start_epoch, args.epochs):
            seed_all(args.seed + epoch * 100003 + rank)
            generator.manual_seed(args.seed + epoch * 100003 + rank)
            if sampler is not None:
                sampler.set_epoch(epoch)
            lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
            optimizer.param_groups[0]["lr"] = args.depth_lr * lr_factor
            if grasp_params:
                optimizer.param_groups[1]["lr"] = args.grasp_lr * lr_factor
            warmup = min(1.0, (epoch + 1) / max(args.relative_warmup_epochs, 1))
            module.set_training(True)
            totals = torch.zeros(6, device=device, dtype=torch.float64)
            for step, batch in enumerate(train_loader):
                if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                    break
                batch = move_batch(batch, device)
                pair_seed = args.seed + epoch * 1000003 + int(batch["dataset_idx"][0])
                optimizer.zero_grad(set_to_none=True)
                result = network(batch, pair_seed, warmup)
                bad = (~torch.isfinite(result["loss"])).to(torch.int32)
                if world_size > 1:
                    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
                if int(bad):
                    raise FloatingPointError("Non-finite training loss; no optimizer step was taken.")
                result["loss"].backward()
                depth_norm, grasp_norm = gradient_norm(depth_params), gradient_norm(grasp_params)
                bad = ((~torch.isfinite(depth_norm)).to(device) | (~torch.isfinite(grasp_norm)).to(device)).to(torch.int32)
                if world_size > 1:
                    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
                if int(bad):
                    raise FloatingPointError("Non-finite gradients; no optimizer step was taken on any rank.")
                if args.clip_mode == "global":
                    torch.nn.utils.clip_grad_norm_(depth_params + grasp_params, args.max_grad_norm, error_if_nonfinite=True)
                else:
                    torch.nn.utils.clip_grad_norm_(depth_params, args.max_grad_norm, error_if_nonfinite=True)
                    if grasp_params:
                        torch.nn.utils.clip_grad_norm_(grasp_params, args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                n = batch["img"].shape[0]
                totals[:4] += torch.stack([result[k].detach() for k in ("loss", "task", "absolute", "relative")]).double() * n
                totals[4] += n
                totals[5] += result["pairs"]["valid"].sum()
                if rank == 0 and (step % args.log_every == 0):
                    record = {"epoch": epoch, "step": step, "variant": args.variant,
                              **{k: float(result[k].detach()) for k in ("loss", "task", "absolute", "relative")},
                              "depth_grad_norm_before_clip": float(depth_norm),
                              "grasp_grad_norm_before_clip": float(grasp_norm), "relative_multiplier": warmup,
                              "metrics_timing": "prediction before this optimizer step",
                              **depth_metrics(result["prediction"], batch["gt_depth_m"], cfg),
                              **pair_metrics(result["prediction"], batch["gt_depth_m"], result["pairs"], cfg, batch["K"])}
                    append_jsonl(output / "train_steps.jsonl", record)
                    print(f"epoch={epoch} step={step} loss={record['loss']:.5f} depth_mae={record['mae_m']} pairs={record['pair_count']}", flush=True)
            if world_size > 1:
                dist.all_reduce(totals)
            if int(totals[4]) == 0 or (args.variant != "none" and int(totals[5]) == 0):
                raise RuntimeError("No training samples or no valid relative-depth pairs; inspect GT/camera/cache contracts.")

            module.set_training(False)
            if world_size > 1:
                # Validate the same running statistics on every shard and save
                # exactly that model. Unwrapped evaluation has no DDP forwards.
                for buffer in module.buffers():
                    dist.broadcast(buffer, src=0)
            validation = torch.zeros(4, dtype=torch.float64, device=device)
            # The same GT pairs are regenerated for each frame in every epoch,
            # independent of the train sampler, rank and evaluation batch size.
            with torch.no_grad():
                for batch in eval_loader:
                    batch = move_batch(batch, device)
                    # Evaluate one frame at a time to make validation pair RNG
                    # independent of batching/world size.
                    for b in range(batch["img"].shape[0]):
                        frame = {key: value[b:b + 1] if torch.is_tensor(value) or isinstance(value, list) else value
                                 for key, value in batch.items()}
                        prediction = module.predict_depth(frame)
                        pairs = matched_depth_pairs(frame, cfg, args.seed + int(frame["dataset_idx"][0]))
                        pm = pair_metrics(prediction, frame["gt_depth_m"], pairs, cfg, frame["K"])
                        dm = depth_metrics(prediction, frame["gt_depth_m"], cfg)
                        if dm["mae_m"] is not None:
                            validation[0] += dm["mae_m"]
                            validation[1] += 1
                        if pm["anchor_relative_mae_m"] is not None:
                            validation[2] += pm["anchor_relative_mae_m"]
                            validation[3] += 1
            if world_size > 1:
                dist.all_reduce(validation)
            if int(validation[1]) == 0 or int(validation[3]) == 0:
                raise RuntimeError("No valid validation geometry; refusing to select a best checkpoint.")
            mae = float(validation[0] / validation[1])
            relative_mae = float(validation[2] / validation[3]) if int(validation[3]) else None
            geometry_score = mae + (relative_mae if relative_mae is not None else 0)
            improved = geometry_score < best
            best = min(best, geometry_score)
            if rank == 0:
                record = {"epoch": epoch, "train": {k: float(totals[i] / totals[4])
                          for i, k in enumerate(("loss", "task", "absolute", "relative"))},
                          "train_pairs": int(totals[5]), "validation_depth_mae_m": mae,
                          "validation_anchor_relative_mae_m": relative_mae,
                          "validation_frames": int(validation[1]), "validation_pair_frames": int(validation[3]),
                          "best_geometry_score": best}
                append_jsonl(output / "epochs.jsonl", record)
                payload = {**checkpoint_metadata(model, args), "epoch": epoch + 1,
                           "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                           "best_validation_geometry": best, "world_size": world_size,
                           "train_indices": train_indices, "eval_indices": eval_indices, "validation_metrics": record}
                save_checkpoint(output / "checkpoint.tar", payload)
                save_checkpoint(output / f"epoch_{epoch:02d}.tar", payload)
                if improved:
                    save_checkpoint(output / "best_geometry.tar", payload)
                print(f"epoch={epoch} validation_depth_mae={mae:.6f} anchor_relative_mae={relative_mae}", flush=True)
            if world_size > 1:
                dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
