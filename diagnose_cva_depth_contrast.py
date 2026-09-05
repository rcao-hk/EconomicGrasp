"""Scan predicted-depth contrast with dynamic queries and frozen GT queries.

Outputs are supervised-loss diagnostics, NOT GraspNet AP or force-closure
labels. The fixed protocol replays literal GT-anchor positions, selected views,
both label passes and patch grids. No mesh/DexNet evaluator is called.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cva_depth_geometry import (contrast_depth, depth_b1hw, depth_metrics,
                                matched_depth_pairs, pair_metrics, visible_anchor_pool)
from cva_depth_experiment import (CONTRACT_VERSION, FixedSupportReplay,
                                  add_common_arguments, append_jsonl, capture_depth_outputs,
                                  fixed_anchor_seeds, grasp_objective, load_model, make_dataset,
                                  move_batch, pair_config, replay_depth, seed_all, seed_worker,
                                  source_revision, tensor_copy, validate_output_dir, write_json)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--split", choices=("train", "test_seen", "test_similar", "test_novel"), default="test_seen")
    parser.add_argument("--max_frames", type=int, default=32, help="Evenly selected across the split; 0 means all.")
    parser.add_argument("--betas", type=float, nargs="+", default=[0, 0.25, 0.5, 1, 1.25])
    parser.add_argument("--fixed_queries", type=int, default=256)
    parser.add_argument("--save_depth_frames", type=int, default=4)
    parser.add_argument("--probe_geometry_gradient", action="store_true",
                        help="Optional fixed-query input-Jacobian/finite-difference probe; never updates model parameters.")
    args = parser.parse_args(argv)
    if 1.0 not in args.betas or len(set(args.betas)) != len(args.betas) or any(b < 0 or not np.isfinite(b) for b in args.betas):
        parser.error("--betas must be distinct, finite, nonnegative and contain 1.")
    if args.fixed_queries < 1 or args.save_depth_frames < 0:
        parser.error("fixed_queries must be positive and save_depth_frames nonnegative.")
    return args


def label_metrics(ep):
    valid = ep["batch_grasp_cdf_valid_mask"].bool()
    positive = valid & (ep["batch_grasp_cdf_bins_angle_depth"] > 0)
    distance = (ep["xyz_graspable"] - ep["batch_grasp_point"]).norm(dim=-1)
    return {"queries": int(ep["xyz_graspable"].shape[1]), "valid_candidates": int(valid.sum()),
            "candidate_capacity": valid.numel(), "positive_candidates": int(positive.sum()),
            "cdf_bce_denominator": int(valid.sum()) * int(ep["grasp_cdf_pred_angle_depth"].shape[1]),
            "valid_width_labels": int(ep["batch_grasp_width_valid_mask_angle_depth"].sum()),
            "mean_anchor_match_distance_m": float(distance.mean()),
            "depth_valid_fraction": float(ep["D: DepthValid ratio"])}


def run_frame(model, batch):
    ep = dict(batch)
    ep["cva_force_process_grasp_labels"] = True
    ep["cva_compute_diagnostics"] = False
    ep["rgb_geometry_compute_diagnostics"] = False
    return model(ep)


def measure(ep, depth, batch, pairs, cfg, cfgs):
    task, terms = grasp_objective(ep, cfgs)
    values = {"grasp_total_loss": float(task), **{f"loss_{k}": float(v) for k, v in terms.items()}}
    values.update(label_metrics(ep))
    values.update({f"depth_{k}": v for k, v in depth_metrics(depth, batch["gt_depth_m"], cfg).items()})
    values.update({f"foreground_{k}": v for k, v in depth_metrics(
        depth, batch["gt_depth_m"], cfg, batch["objectness_label_tok"]).items()})
    values.update(pair_metrics(depth, batch["gt_depth_m"], pairs, cfg, batch["K"]))
    if not all(np.isfinite(v) for v in values.values() if v is not None):
        raise FloatingPointError("Non-finite diagnostic metric.")
    return values


def summarize(rows):
    summaries = []
    for protocol in sorted({r["protocol"] for r in rows}):
        chosen = [r for r in rows if r["protocol"] == protocol]
        baseline = {r["dataset_idx"]: r for r in chosen if r["beta"] == 1}
        for beta in sorted({r["beta"] for r in chosen}):
            group = [r for r in chosen if r["beta"] == beta]
            stats = {"protocol": protocol, "beta": beta, "frames": len(group),
                     "scenes": len({r["scene_idx"] for r in group})}
            for key in ("grasp_total_loss", "loss_cdf", "loss_view", "loss_width",
                        "valid_candidates", "positive_candidates", "mean_anchor_match_distance_m",
                        "depth_mae_m", "foreground_mae_m", "depth_std_ratio", "anchor_relative_mae_m"):
                values = [r[key] for r in group if r.get(key) is not None]
                deltas = [r[key] - baseline[r["dataset_idx"]][key] for r in group
                          if r.get(key) is not None and baseline[r["dataset_idx"]].get(key) is not None]
                stats[key + "_mean"] = float(np.mean(values)) if values else None
                stats[key + "_paired_delta_vs_beta1"] = float(np.mean(deltas)) if deltas else None
            summaries.append(stats)
    return summaries


def probe_gradient(model, batch, cached_depth, pred, replay, cfgs):
    """Partial input sensitivity with fixed physical queries; not a DPT update."""
    gse = model.spatial_enhancer
    group_cfg = model.kview_grasp_module.group.config
    old_gse, old_support = gse.detach_depth_grad, group_cfg.detach_depth
    gse.detach_depth_grad = False
    group_cfg.detach_depth = False
    try:
        variable = pred.detach().clone().requires_grad_(True)
        replay.begin()
        with torch.enable_grad(), replay_depth(model, cached_depth, variable):
            ep = run_frame(model, batch)
            loss, _ = grasp_objective(ep, cfgs)
            gradient, = torch.autograd.grad(loss, variable)
        replay.finish()
        direction = pred - pred.mean((1, 2, 3), keepdim=True)
        slope = float((gradient * direction).sum())
        epsilon = 1e-3
        losses = []
        for sign in (-1, 1):
            replay.begin()
            with torch.no_grad(), replay_depth(model, cached_depth, pred + sign * epsilon * direction):
                ep = run_frame(model, batch)
                value, _ = grasp_objective(ep, cfgs)
            replay.finish()
            losses.append(float(value))
        finite_difference = (losses[1] - losses[0]) / (2 * epsilon)
        gt = depth_b1hw(batch["gt_depth_m"]).to(pred)
        valid = torch.isfinite(gt) & (gt >= model.min_depth) & (gt <= model.max_depth)
        alignment = float((gradient * torch.where(valid, pred - torch.nan_to_num(gt), 0.0)).sum())
        return {"input_gradient_norm": float(gradient.norm()), "contrast_slope_autograd": slope,
                "contrast_slope_finite_difference": finite_difference,
                "slope_relative_discrepancy": abs(slope - finite_difference) / max(abs(slope), abs(finite_difference), 1e-6),
                "gradient_dot_depth_error": alignment,
                "scope": "fixed_GT_query_GSE_and_support_depth_values_only"}
    finally:
        gse.detach_depth_grad, group_cfg.detach_depth = old_gse, old_support


def save_depth_figure(path, pred, gt, betas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arrays = [gt[0, 0].cpu().numpy(), pred[0, 0].cpu().numpy()]
    arrays += [contrast_depth(pred, beta)[0, 0].cpu().numpy() for beta in betas]
    titles = ["GT (m)", "Prediction (m)"] + [f"beta={beta:g}" for beta in betas]
    limits = np.nanpercentile(np.concatenate([a[np.isfinite(a) & (a > 0)] for a in arrays]), [1, 99])
    figure, axes = plt.subplots(1, len(arrays), figsize=(3 * len(arrays), 3), squeeze=False)
    for axis, array, title in zip(axes[0], arrays, titles):
        axis.imshow(array, vmin=limits[0], vmax=limits[1], cmap="magma")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    cfg = pair_config(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Full-model diagnosis needs the repository's CUDA environment. Run test_cva_depth_geometry.py for CPU tests.")
    seed_all(args.seed)
    model, _, cfgs = load_model(args, torch.device("cuda"), training=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    from dataset.graspnet_dataset import collate_fn
    from models.p0b_topk_exact_override import (P0B_EXACT_QUERY_VIEW_OVERRIDE_KEY,
                                               install_p0b_exact_query_selector_override)
    install_p0b_exact_query_selector_override(model)
    dataset, indices = make_dataset(args, args.split, args.max_frames)
    validate_output_dir(args.output_dir)
    output_dir = Path(args.output_dir)
    write_json(output_dir / "contract.json", {"version": CONTRACT_VERSION, "source_revision": source_revision(),
        "args": vars(args), "pair_config": cfg.to_dict(), "dataset_indices": indices,
        "view_policy": "deterministic top1; fixed views use GT view-graspness argmax at visible GT anchors",
        "depth_mean": "per-image full predicted map, held constant for the scan",
        "depth_clipping": False, "geometry_detach_during_scan": True,
        "metric": "supervised loss and geometry diagnostics; not AP", "dexnet_calls": 0})
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_fn, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(args.seed))
    rows, skipped = [], []
    for frame_number, batch in enumerate(loader):
        batch = move_batch(batch, torch.device("cuda"))
        identity = {k: int(batch[k].item()) for k in ("scene_idx", "anno_idx", "dataset_idx")}
        with torch.no_grad(), capture_depth_outputs(model) as captured:
            baseline = run_frame(model, batch)
        if len(captured) != 1:
            raise RuntimeError("Expected one metric-depth forward per frame.")
        cached_depth = captured[0]
        pred = depth_b1hw(cached_depth[0]).detach()
        pairs = matched_depth_pairs(batch, cfg, args.seed + identity["dataset_idx"])
        with torch.no_grad():
            for beta in args.betas:
                depth = contrast_depth(pred, beta)
                with replay_depth(model, cached_depth, depth):
                    ep = run_frame(model, batch)
                if beta == 1:
                    torch.testing.assert_close(ep["grasp_cdf_pred_angle_depth"], baseline["grasp_cdf_pred_angle_depth"], rtol=1e-5, atol=1e-6)
                    if not torch.equal(ep["token_sel_idx"], baseline["token_sel_idx"]):
                        raise RuntimeError("beta=1 replay changed autonomous query selection.")
                row = {**identity, "protocol": "dynamic", "beta": beta, **measure(ep, depth, batch, pairs, cfg, cfgs)}
                rows.append(row)
                append_jsonl(output_dir / "frames.jsonl", row)

        pool = visible_anchor_pool(batch, 0, cfg)
        count = min(args.fixed_queries, len(pool["pixel"]))
        if not count:
            skipped.append({**identity, "reason": "no_visible_GT_anchor"})
        else:
            take = torch.linspace(0, len(pool["pixel"]) - 1, count, device=pred.device).round().long()
            xyz, pixels = pool["xyz"][take][None], pool["pixel"][take][None]
            with fixed_anchor_seeds(model, xyz, pixels):
                gt = depth_b1hw(batch["gt_depth_m"]).to(pred)
                with torch.no_grad(), replay_depth(model, cached_depth, gt):
                    reference = run_frame(model, batch)
                views = reference["batch_grasp_view_graspness"].argmax(-1).long()
                fixed_batch = dict(batch)
                fixed_batch[P0B_EXACT_QUERY_VIEW_OVERRIDE_KEY] = views[..., None]
                with FixedSupportReplay(model, views) as replay:
                    replay.begin(recording=True)
                    with torch.no_grad(), replay_depth(model, cached_depth, gt):
                        reference = run_frame(model, fixed_batch)
                    replay.finish()
                    fingerprint = replay.fingerprint()
                    if not bool(reference["batch_grasp_cdf_valid_mask"].any()):
                        skipped.append({**identity, "reason": "GT_anchor_views_have_no_valid_CDF_labels"})
                    else:
                        for beta in args.betas:
                            depth = contrast_depth(pred, beta)
                            replay.begin()
                            with torch.no_grad(), replay_depth(model, cached_depth, depth):
                                ep = run_frame(model, fixed_batch)
                            replay.finish()
                            for key in ("xyz_graspable", "token_sel_idx", "grasp_top_view_inds", "batch_grasp_cdf_valid_mask"):
                                if not torch.equal(ep[key], reference[key]):
                                    raise RuntimeError(f"Fixed protocol changed {key}.")
                            row = {**identity, "protocol": "fixed_gt", "beta": beta,
                                   "reference_sha256": fingerprint, **measure(ep, depth, batch, pairs, cfg, cfgs)}
                            rows.append(row)
                            append_jsonl(output_dir / "frames.jsonl", row)
                        if args.probe_geometry_gradient:
                            probe = probe_gradient(model, fixed_batch, cached_depth, pred, replay, cfgs)
                            append_jsonl(output_dir / "gradient_probe.jsonl", {**identity, **probe})
        if frame_number < args.save_depth_frames:
            save_depth_figure(output_dir / f"depth_scene{identity['scene_idx']:04d}_{identity['anno_idx']:04d}.png",
                              pred, depth_b1hw(batch["gt_depth_m"]), args.betas)
        print(f"[{frame_number + 1}/{len(dataset)}] scene={identity['scene_idx']} anno={identity['anno_idx']} fixed_anchors={count}", flush=True)
    summaries = summarize(rows)
    write_json(output_dir / "summary.json", {"rows": summaries, "skipped_fixed_frames": skipped,
        "comparison": "paired beta-minus-one within each protocol; absolute losses across protocols have different query distributions"})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved {len(rows)} diagnostic rows to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
