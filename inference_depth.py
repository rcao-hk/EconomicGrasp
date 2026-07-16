#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset.graspnet_dataset import GraspNetDataset, GraspNetMultiDataset, collate_fn


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def move_batch_to_device(batch_data, device):
    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        elif 'graph' in key:
            for i in range(len(batch_data[key])):
                batch_data[key][i] = batch_data[key][i].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
    return batch_data


def ensure_bhw(depth_tensor, name="tensor"):
    """
    Convert tensor to shape (B, H, W).
    Accepts:
      (B, H, W)
      (B, 1, H, W)
      (H, W)
    """
    if not torch.is_tensor(depth_tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(depth_tensor)}")

    if depth_tensor.ndim == 4 and depth_tensor.shape[1] == 1:
        depth_tensor = depth_tensor[:, 0]
    elif depth_tensor.ndim == 3:
        pass
    elif depth_tensor.ndim == 2:
        depth_tensor = depth_tensor.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported {name} shape: {tuple(depth_tensor.shape)}")
    return depth_tensor


def resize_pred_to_gt(pred_depth, gt_depth):
    """
    pred_depth: (B,H,W)
    gt_depth:   (B,H,W)
    """
    if pred_depth.shape[-2:] == gt_depth.shape[-2:]:
        return pred_depth
    pred_depth = F.interpolate(
        pred_depth.unsqueeze(1),
        size=gt_depth.shape[-2:],
        mode="bilinear",
        align_corners=False
    ).squeeze(1)
    return pred_depth


def extract_depth_prediction(end_points):
    """
    Fixed key according to your model forward.
    """
    if not isinstance(end_points, dict):
        raise TypeError(f"Expected end_points to be dict, got {type(end_points)}")

    if "depth_map_pred" not in end_points:
        tensor_keys = [(k, tuple(v.shape)) for k, v in end_points.items() if torch.is_tensor(v)]
        raise KeyError(
            'Cannot find "depth_map_pred" in end_points. '
            f"Available tensor keys: {tensor_keys}"
        )

    return ensure_bhw(end_points["depth_map_pred"], name="depth_map_pred")


def _reshape_flat_mask_to_hw(mask_flat, target_hw):
    """
    mask_flat: (B, N)
    target_hw: (H, W) of target depth map

    Priority:
      1) if N == H*W -> reshape directly
      2) if sqrt(N) is integer -> reshape to square token map
      3) otherwise raise error
    """
    B, N = mask_flat.shape
    H, W = target_hw

    if N == H * W:
        return mask_flat.reshape(B, H, W)

    s = int(round(math.sqrt(N)))
    if s * s == N:
        return mask_flat.reshape(B, s, s)

    raise ValueError(
        f"Cannot reshape flat objectness_label_tok of shape {(B, N)} "
        f"to 2D map. target_hw={target_hw}, N is not H*W and not a square number."
    )


def extract_object_mask_from_batch(batch_data, target_hw):
    """
    Return object foreground mask in shape (B, H, W), bool.

    Supported input shapes for batch_data['objectness_label_tok']:
      - (B, 1, Ht, Wt)
      - (B, Ht, Wt)
      - (B, Ntok)
      - (B, Ntok, 1)
      - (B, 1, Ntok)

    Then upsample to target_hw with nearest interpolation.
    """
    if "objectness_label_tok" not in batch_data:
        raise KeyError("batch_data does not contain 'objectness_label_tok'")

    obj = batch_data["objectness_label_tok"]
    if not torch.is_tensor(obj):
        raise TypeError(f"objectness_label_tok must be tensor, got {type(obj)}")

    if obj.ndim == 4:
        if obj.shape[1] != 1:
            raise ValueError(f"Unsupported 4D objectness_label_tok shape: {tuple(obj.shape)}")
        obj_map = obj[:, 0].float()  # (B,Ht,Wt)

    elif obj.ndim == 3:
        # (B,Ht,Wt)
        if obj.shape[-1] != 1 and obj.shape[1] != 1:
            obj_map = obj.float()
        # (B,Ntok,1)
        elif obj.shape[-1] == 1:
            obj_map = _reshape_flat_mask_to_hw(obj[..., 0].float(), target_hw)
        # (B,1,Ntok)
        elif obj.shape[1] == 1:
            obj_map = _reshape_flat_mask_to_hw(obj[:, 0].float(), target_hw)
        else:
            raise ValueError(f"Unsupported 3D objectness_label_tok shape: {tuple(obj.shape)}")

    elif obj.ndim == 2:
        obj_map = _reshape_flat_mask_to_hw(obj.float(), target_hw)

    else:
        raise ValueError(f"Unsupported objectness_label_tok shape: {tuple(obj.shape)}")

    if obj_map.shape[-2:] != target_hw:
        obj_map = F.interpolate(
            obj_map.unsqueeze(1),
            size=target_hw,
            mode="nearest"
        ).squeeze(1)

    return obj_map > 0


def compute_depth_sums(pred, gt, min_depth=None, max_depth=None, extra_mask=None):
    """
    pred, gt: (B,H,W), torch.float
    extra_mask: optional bool mask (B,H,W), e.g. object foreground mask

    Returns pixel-wise aggregated sums for overall metric computation.
    """
    pred = pred.float()
    gt = gt.float()

    if min_depth is not None or max_depth is not None:
        lo = -float("inf") if min_depth is None else float(min_depth)
        hi =  float("inf") if max_depth is None else float(max_depth)
        pred = pred.clamp(min=lo, max=hi)

    valid = torch.isfinite(gt) & torch.isfinite(pred) & (gt > 0)
    if min_depth is not None:
        valid = valid & (gt >= float(min_depth))
    if max_depth is not None:
        valid = valid & (gt <= float(max_depth))

    fg_pixels = None
    fg_ratio = None
    if extra_mask is not None:
        if extra_mask.shape != gt.shape:
            raise ValueError(
                f"extra_mask shape {tuple(extra_mask.shape)} != gt shape {tuple(gt.shape)}"
            )
        extra_mask = extra_mask.bool()
        fg_pixels = int(extra_mask.sum().item())
        fg_ratio = float(extra_mask.float().mean().item())
        valid = valid & extra_mask

    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return {
            "valid_pixels": 0,
            "fg_pixels": fg_pixels,
            "fg_ratio": fg_ratio,
            "sum_abs": 0.0,
            "sum_sq": 0.0,
            "sum_rel": 0.0,
            "sum_delta1": 0.0,
            "sum_delta2": 0.0,
            "sum_delta3": 0.0,
        }

    pred_v = pred[valid]
    gt_v = gt[valid]

    abs_err = torch.abs(pred_v - gt_v)
    sq_err = (pred_v - gt_v) ** 2
    rel_err = abs_err / torch.clamp(gt_v, min=1e-8)

    ratio = torch.maximum(
        pred_v / torch.clamp(gt_v, min=1e-8),
        gt_v / torch.clamp(pred_v, min=1e-8)
    )

    delta1 = (ratio < 1.25).float()
    delta2 = (ratio < 1.25 ** 2).float()
    delta3 = (ratio < 1.25 ** 3).float()

    return {
        "valid_pixels": n_valid,
        "fg_pixels": fg_pixels,
        "fg_ratio": fg_ratio,
        "sum_abs": float(abs_err.sum().item()),
        "sum_sq": float(sq_err.sum().item()),
        "sum_rel": float(rel_err.sum().item()),
        "sum_delta1": float(delta1.sum().item()),
        "sum_delta2": float(delta2.sum().item()),
        "sum_delta3": float(delta3.sum().item()),
    }


def sums_to_metrics(sums):
    n = max(int(sums["valid_pixels"]), 1)
    return {
        "valid_pixels": int(sums["valid_pixels"]),
        "fg_pixels": None if sums["fg_pixels"] is None else int(sums["fg_pixels"]),
        "fg_ratio": None if sums["fg_ratio"] is None else float(sums["fg_ratio"]),
        "mae": sums["sum_abs"] / n,
        "rmse": (sums["sum_sq"] / n) ** 0.5,
        "rel": sums["sum_rel"] / n,
        "delta1": sums["sum_delta1"] / n,
        "delta2": sums["sum_delta2"] / n,
        "delta3": sums["sum_delta3"] / n,
    }


def add_sums(dst, src):
    for k in ["valid_pixels", "sum_abs", "sum_sq", "sum_rel", "sum_delta1", "sum_delta2", "sum_delta3"]:
        dst[k] += src[k]

    if src["fg_pixels"] is not None:
        if dst["fg_pixels"] is None:
            dst["fg_pixels"] = 0
        dst["fg_pixels"] += src["fg_pixels"]

    if src["fg_ratio"] is not None:
        if dst["fg_ratio_sum"] is None:
            dst["fg_ratio_sum"] = 0.0
            dst["fg_ratio_count"] = 0
        dst["fg_ratio_sum"] += src["fg_ratio"]
        dst["fg_ratio_count"] += 1


def build_dataset(args):
    load_label = bool(args.object_only)

    if args.multi_modal:
        dataset = GraspNetMultiDataset(
            args.dataset_root,
            split=args.test_mode,
            camera=args.camera,
            num_points=args.num_point,
            remove_outlier=True,
            augment=False,
            load_label=load_label
        )
    else:
        dataset = GraspNetDataset(
            args.dataset_root,
            split=args.test_mode,
            camera=args.camera,
            num_points=args.num_point,
            remove_outlier=True,
            augment=False,
            load_label=load_label
        )
    return dataset


def build_eval_subset(dataset, sample_interval):
    """
    sample_interval:
      - 1.0: use full dataset
      - 0.1: use about 10% of dataset, evenly sampled
    """
    total = len(dataset)

    if sample_interval <= 0:
        raise ValueError(f"sample_interval must be > 0, got {sample_interval}")

    if sample_interval >= 1.0:
        indices = list(range(total))
        return dataset, indices

    num_keep = max(1, int(round(total * sample_interval)))
    indices = np.linspace(0, total - 1, num=num_keep, dtype=np.int64).tolist()
    subset = Subset(dataset, indices)
    return subset, indices


def build_dataloader(args):
    full_dataset = build_dataset(args)
    eval_dataset, sampled_indices = build_eval_subset(full_dataset, args.sample_interval)

    test_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=my_worker_init_fn,
        collate_fn=collate_fn
    )
    return full_dataset, eval_dataset, test_dataloader, sampled_indices


def build_model(method_name, args, device):
    if method_name == "c1":
        from models.economicgrasp_depth_c1 import economicgrasp_c1
        net = economicgrasp_c1(
            depth_stride=2,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            is_training=False,
            vis_dir=args.save_root,
            vis_every=10,
        )
    elif method_name == "c2_3":
        from models.economicgrasp_depth_c1 import economicgrasp_c2_3
        net = economicgrasp_c2_3(
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            is_training=False,
            vis_dir=args.save_root,
            vis_every=10,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    net.to(device)
    return net


def load_checkpoint_to_model(net, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    try:
        net.load_state_dict(checkpoint["model_state_dict"])
    except Exception:
        net.load_state_dict(checkpoint)
    print(f"-> loaded checkpoint: {ckpt_path}")


@torch.no_grad()
def evaluate_one_method(method_name, ckpt_path, dataloader, sampled_indices, device, args):
    net = build_model(method_name, args, device)
    load_checkpoint_to_model(net, ckpt_path)
    net.eval()

    global_sums = {
        "valid_pixels": 0,
        "fg_pixels": None,
        "fg_ratio_sum": None,
        "fg_ratio_count": 0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "sum_rel": 0.0,
        "sum_delta1": 0.0,
        "sum_delta2": 0.0,
        "sum_delta3": 0.0,
    }

    per_batch = []
    tic = time.time()
    processed_samples = 0

    for batch_idx, batch_data in enumerate(dataloader):
        batch_data = move_batch_to_device(batch_data, device)

        end_points = net(batch_data)

        pred_depth = extract_depth_prediction(end_points)   # (B,H,W)
        gt_depth = ensure_bhw(batch_data["gt_depth_m"], name="gt_depth_m")
        pred_depth = resize_pred_to_gt(pred_depth, gt_depth)

        obj_mask = None
        if args.object_only:
            obj_mask = extract_object_mask_from_batch(batch_data, target_hw=gt_depth.shape[-2:])

        batch_sums = compute_depth_sums(
            pred_depth,
            gt_depth,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            extra_mask=obj_mask
        )
        batch_metrics = sums_to_metrics(batch_sums)

        bs = int(gt_depth.shape[0])
        start = processed_samples
        end = processed_samples + bs
        batch_sample_indices = sampled_indices[start:end]
        processed_samples += bs

        batch_metrics["batch_idx"] = int(batch_idx)
        batch_metrics["batch_size"] = bs
        batch_metrics["sample_indices"] = [int(x) for x in batch_sample_indices]

        per_batch.append(batch_metrics)
        add_sums(global_sums, batch_sums)

        if batch_idx % args.print_every == 0:
            elapsed = time.time() - tic
            if args.object_only:
                print(
                    f"[{method_name}] batch {batch_idx:04d} | "
                    f"mae={batch_metrics['mae']:.6f}, "
                    f"rmse={batch_metrics['rmse']:.6f}, "
                    f"valid={batch_metrics['valid_pixels']}, "
                    f"fg_ratio={batch_metrics['fg_ratio']:.4f} | "
                    f"time={elapsed:.2f}s"
                )
            else:
                print(
                    f"[{method_name}] batch {batch_idx:04d} | "
                    f"mae={batch_metrics['mae']:.6f}, "
                    f"rmse={batch_metrics['rmse']:.6f}, "
                    f"valid={batch_metrics['valid_pixels']} | "
                    f"time={elapsed:.2f}s"
                )
            tic = time.time()

    denom = max(1, global_sums["valid_pixels"])
    overall = {
        "valid_pixels": int(global_sums["valid_pixels"]),
        "fg_pixels": None if global_sums["fg_pixels"] is None else int(global_sums["fg_pixels"]),
        "fg_ratio": None,
        "mae": global_sums["sum_abs"] / denom,
        "rmse": (global_sums["sum_sq"] / denom) ** 0.5,
        "rel": global_sums["sum_rel"] / denom,
        "delta1": global_sums["sum_delta1"] / denom,
        "delta2": global_sums["sum_delta2"] / denom,
        "delta3": global_sums["sum_delta3"] / denom,
    }
    if global_sums["fg_ratio_sum"] is not None and global_sums["fg_ratio_count"] > 0:
        overall["fg_ratio"] = global_sums["fg_ratio_sum"] / global_sums["fg_ratio_count"]

    mean_batch_mae = float(np.mean([x["mae"] for x in per_batch])) if len(per_batch) > 0 else None

    return {
        "method": method_name,
        "checkpoint": ckpt_path,
        "overall": overall,
        "mean_batch_mae": mean_batch_mae,
        "num_batches": len(per_batch),
        "num_samples": len(sampled_indices),
        "per_batch": per_batch,
    }


def main():
    parser = argparse.ArgumentParser("Compare depth estimation of C1 and C2.3 on GraspNet-1B")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--camera", type=str, default="realsense")
    parser.add_argument("--test_mode", type=str, default="test_seen")
    parser.add_argument("--num_point", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--multi_modal", action="store_true")

    parser.add_argument("--min_depth", type=float, default=0.2)
    parser.add_argument("--max_depth", type=float, default=1.0)

    parser.add_argument("--sample_interval", type=float, default=1.0,
                        help="Fraction of dataset to evaluate. 1.0=all, 0.1=10% evenly sampled.")
    parser.add_argument("--object_only", action="store_true",
                        help="If set, compute depth metrics only on object foreground area from objectness_label_tok.")

    parser.add_argument("--c1_ckpt", type=str, required=True)
    parser.add_argument("--c23_ckpt", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--save_root", type=str, required=True)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_root, exist_ok=True)

    full_dataset, eval_dataset, dataloader, sampled_indices = build_dataloader(args)

    print(f"Full dataset size : {len(full_dataset)}")
    print(f"Eval dataset size : {len(eval_dataset)}")
    print(f"sample_interval   : {args.sample_interval}")
    print(f"object_only       : {args.object_only}")

    results = {
        "config": vars(args),
        "dataset_info": {
            "full_dataset_size": len(full_dataset),
            "eval_dataset_size": len(eval_dataset),
            "num_sampled_indices": len(sampled_indices),
            "sampled_indices_preview": [int(x) for x in sampled_indices[:50]],
        },
        "methods": {}
    }

    c1_result = evaluate_one_method("c1", args.c1_ckpt, dataloader, sampled_indices, device, args)
    c23_result = evaluate_one_method("c2_3", args.c23_ckpt, dataloader, sampled_indices, device, args)

    results["methods"]["c1"] = c1_result
    results["methods"]["c2_3"] = c23_result

    results["comparison"] = {
        "overall_mae_gap_c23_minus_c1": c23_result["overall"]["mae"] - c1_result["overall"]["mae"],
        "overall_rmse_gap_c23_minus_c1": c23_result["overall"]["rmse"] - c1_result["overall"]["rmse"],
        "overall_rel_gap_c23_minus_c1": c23_result["overall"]["rel"] - c1_result["overall"]["rel"],
        "mean_batch_mae_gap_c23_minus_c1": (
            c23_result["mean_batch_mae"] - c1_result["mean_batch_mae"]
            if c1_result["mean_batch_mae"] is not None and c23_result["mean_batch_mae"] is not None
            else None
        )
    }

    json_path = os.path.join(args.save_root, f"depth_compare_{args.test_mode}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved json to: {json_path}")

    print("\n========== OVERALL DEPTH METRICS ==========")
    for method_name in ["c1", "c2_3"]:
        m = results["methods"][method_name]["overall"]
        extra = ""
        if args.object_only:
            extra = f", fg_pixels={m['fg_pixels']}, mean_fg_ratio={m['fg_ratio']:.6f}"
        print(
            f"{method_name}: "
            f"MAE={m['mae']:.6f}, "
            f"RMSE={m['rmse']:.6f}, "
            f"REL={m['rel']:.6f}, "
            f"delta1={m['delta1']:.6f}, "
            f"delta2={m['delta2']:.6f}, "
            f"delta3={m['delta3']:.6f}, "
            f"valid_pixels={m['valid_pixels']}"
            f"{extra}"
        )

    print("\n========== OVERALL MAE ==========")
    print(f"C1   overall MAE: {results['methods']['c1']['overall']['mae']:.6f}")
    print(f"C2.3 overall MAE: {results['methods']['c2_3']['overall']['mae']:.6f}")
    print(f"Gap  (C2.3 - C1): {results['comparison']['overall_mae_gap_c23_minus_c1']:.6f}")

    print("\n========== MEAN OF BATCH MAE ==========")
    print(f"C1   mean batch MAE: {results['methods']['c1']['mean_batch_mae']:.6f}")
    print(f"C2.3 mean batch MAE: {results['methods']['c2_3']['mean_batch_mae']:.6f}")
    print(f"Gap  (C2.3 - C1): {results['comparison']['mean_batch_mae_gap_c23_minus_c1']:.6f}")


if __name__ == "__main__":
    main()