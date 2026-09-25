#!/usr/bin/env python3
"""Repeat active CDF matching; contrast its mapping with the old raw scatter.

Accepts the same checkpoint/dataset/config arguments as the dynamics entrypoint.
Use a NEW --output directory. The default reproduces P0 audit batch zero and
uses two cached train probe frames only for Experiment initialization.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def compare_labels(current, reference):
    pairs = {
        "batch_grasp_cdf_bins_angle_depth": "batch_grasp_cdf_valid_mask",
        "batch_grasp_width_angle_depth": "batch_grasp_width_valid_mask_angle_depth",
        "batch_grasp_point": None, "batch_grasp_view_graspness": None,
        "batch_grasp_cdf_valid_mask": None,
        "batch_grasp_width_valid_mask_angle_depth": None,
    }
    result = {}
    for key, mask_key in pairs.items():
        a, b = current[key], reference[key]
        changed = a != b
        row = {"changed": int(changed.sum()), "numel": a.numel(),
               "max_abs": float((a.double() - b.double()).abs().max())}
        if mask_key:
            union = current[mask_key].bool() | reference[mask_key].bool()
            row.update(changed_on_valid_union=int((changed & union).sum()),
                       changed_outside_valid_union=int((changed & ~union).sum()),
                       mask_changed=int((current[mask_key] != reference[mask_key]).sum()))
        result[key] = row
    return result


def inverse_mapping_evidence(ep, labels, repeat):
    """Count duplicate inverse destinations on the *actual* nearest cache rows."""
    torch = labels.torch
    xyz = ep["xyz_graspable"]
    views = labels.generate_grasp_views(int(labels.cfgs.num_view)).to(xyz)
    V = len(views)
    evidence = []
    for b in range(len(xyz)):
        poses = ep["object_poses_list"][b]
        points, ends = [], []
        for i, pose in enumerate(poses):
            raw = ep["grasp_points_list"][b][i].to(xyz)
            points.append(labels.transform_point_cloud(raw, pose.to(xyz), "3x4"))
            ends.append(sum(len(p) for p in points))
        _, nn, _ = labels.knn_points(xyz[b].unsqueeze(0), torch.cat(points).unsqueeze(0), K=1)
        nn = nn.reshape(-1).long()
        owner = torch.bucketize(nn, torch.tensor(ends, device=xyz.device), right=True)
        starts = [0, *ends[:-1]]
        for i, pose in enumerate(poses):
            query_rows = torch.where(owner == i)[0]
            if not len(query_rows):
                continue
            local = nn[query_rows] - starts[i]
            top = ep["top_view_index_list"][b][i].index_select(0, local.cpu()).to(device=xyz.device, dtype=torch.long)
            rotated = labels.transform_point_cloud(views, pose[:3, :3].to(xyz), "3x3")
            _, view_index, _ = labels.knn_points(views.unsqueeze(0), rotated.unsqueeze(0), K=1)
            view_index = view_index.reshape(-1).long()
            multiplicity = torch.bincount(view_index, minlength=V)
            row, slot, scene = torch.where(view_index == top.unsqueeze(-1))
            dest = row * top.shape[1] + slot
            repeated_destinations = int(dest.numel() - dest.unique().numel())
            outcomes = []
            for _ in range(repeat):
                old_inverse = -torch.ones_like(top)
                old_inverse[row, slot] = scene
                outcomes.append(old_inverse)
            varying = [int((x != outcomes[0]).sum()) for x in outcomes[1:]]
            fixed = labels._deterministic_top_view_scene(view_index, top)
            fixed_changes = [int((labels._deterministic_top_view_scene(view_index, top) != fixed).sum())
                             for _ in range(repeat - 1)]
            evidence.append({"batch": b, "object": i, "matched_queries": len(query_rows),
                             "object_views_with_multiple_scene_preimages": int((multiplicity > 1).sum()),
                             "duplicate_destination_extra_writes": repeated_destinations,
                             "ambiguous_topk_destinations": int((multiplicity[top] > 1).sum()),
                             "max_scene_preimages": int(multiplicity.max()),
                             "old_scatter_repeated_inverse_changed_vs_first": varying,
                             "active_max_scene_mapper_repeated_changes": fixed_changes,
                             "old_scatter_vs_active_max_scene_mapper_changes":
                             [int((old != fixed).sum()) for old in outcomes]})
    return evidence


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--label_repeats", type=int, default=10)
    parser.add_argument("--label_indices", default="")
    options, remaining_cli = parser.parse_known_args()
    import torch
    import numpy as np
    import depth_dynamics as dd
    import train_cva_depth_dynamics as trainer
    args, original_args = trainer.parse_args(remaining_cli)
    args.train_probe_frames, args.heldout_test_probe_frames = 2, 0
    args.directional_batches, args.audit_batches = 0, 1
    trainer.torch, trainer.np, trainer.dd = torch, np, dd
    trainer.ORIGINAL_COMMAND = [sys.executable, *sys.argv]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    experiment = None
    try:
        experiment = trainer.Experiment(args, original_args)
        import utils.label_generation as labels
        indices = ([int(x) for x in options.label_indices.split(",")]
                   if options.label_indices else np.linspace(0, len(experiment.stream.dataset) - 1,
                                                             experiment.cfg.batch_size, dtype=int).tolist())
        batch = experiment.original.collate_fn([experiment.stream.sample(i, epoch=0) for i in indices])
        report = {"indices": indices, "repeats": options.label_repeats,
                  "git": experiment.contract["git"], "routes": {},
                  "active_label_mapper": "deterministic_max_scene_index",
                  "inverse_demo": "old conflicting scatter is tested separately; not used by active matcher"}
        forward_reference = None
        for route in ("none", "all"):
            experiment.model.set_depth_grad_routes(route)
            with experiment.diagnostic_context(train_mode=True, seed=trainer.stable_seed(args.seed, "audit", 0)):
                # Match P0's grad-enabled model execution, then discard the
                # network graph. Only label inputs/outputs are needed below.
                loss, ep = experiment.forward(batch, capture=False)
                input_keys = {"xyz_graspable", "grasp_top_view_inds", "object_poses_list",
                              "grasp_points_list", "view_graspness_list", "top_view_index_list",
                              "grasp_cdf_bins_list", "grasp_widths_depth_list",
                              "grasp_width_valids_depth_list", "cdf_thresholds", "batch_valid_mask"}
                fixed = {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in ep.items()
                         if k in input_keys or k.startswith("batch_grasp_")}
                del ep, loss
                with torch.no_grad():
                    if forward_reference is None:
                        forward_reference = fixed
                    route_report = {"model_forward_vs_none": compare_labels(fixed, forward_reference),
                                    "inverse_mapping": inverse_mapping_evidence(fixed, labels, options.label_repeats),
                                    "repeat_comparisons": []}
                    reference = None
                    for i in range(options.label_repeats):
                        _, current = labels.process_grasp_labels_cdf_width(dict(fixed))
                        current = {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in current.items()}
                        if reference is None:
                            reference = current
                        route_report["repeat_comparisons"].append({"repeat": i,
                            "vs_first": compare_labels(current, reference),
                            "vs_model_forward": compare_labels(current, fixed)})
                    report["routes"][route] = route_report
                    dd.write_json(experiment.diag / "cdf_label_determinism.json", report)
                    print(f"[label determinism] route={route}, report={experiment.diag / 'cdf_label_determinism.json'}", flush=True)
        print("Completed active-label repeats and separate old-scatter demonstration.", flush=True)
    finally:
        if experiment is not None:
            experiment.trainer.close()


if __name__ == "__main__":
    main()
