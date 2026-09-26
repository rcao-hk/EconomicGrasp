#!/usr/bin/env python3
"""Summarize measured CVA depth trajectories without inferring a mechanism."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Incomplete or invalid JSON at {path}:{number}") from exc
    return rows


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def avg(values):
    values = [v for v in values if finite(v)]
    return mean(values) if values else None


def flatten_metrics(record):
    """Accept one per-image record or an explicitly nested metric list."""
    if "regions" in record:
        return [record]
    for key in ("depth_metrics", "metrics", "images", "depth"):
        value = record.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and "regions" in value:
            return [value]
    return []


LOSS_TERMS = ("depth", "objectness", "graspness", "view", "cdf", "width")
COVERAGE_PANELS = (
    ("D: DepthValid ratio", "Predicted-depth valid fraction"),
    ("D: Dynamics point valid fraction", "Query point-valid fraction"),
    ("D: Dynamics view valid fraction", "Query view-valid fraction"),
    ("D: Dynamics CDF valid count", "CDF valid candidates"),
    ("D: Dynamics CDF positive count", "CDF positive candidates"),
    ("D: Dynamics width valid count", "Width valid candidates"),
    ("D: Dynamics seed fallback fraction", "Seed fallback fraction"),
    ("D: Dynamics repeated seed fraction", "Repeated-seed fraction"),
    ("D: Dynamics NN distance mean m", "Mean nearest-point distance (m)"),
)
SWITCH_PANELS = (
    ("query_identity_switch_rate_aligned_slots", "Query identity switches (aligned slots)"),
    ("nn_switch_rate_aligned_slots", "Nearest-point switches (aligned slots)"),
    ("nn_switch_rate_same_identity", "Nearest-point switches (same query identity)"),
    ("point_valid_switch_rate_aligned_slots", "Point-valid mask switches (aligned slots)"),
    ("view_valid_switch_rate_aligned_slots", "View-valid mask switches (aligned slots)"),
    ("same_identity_query_displacement_m", "Same-query displacement (m)"),
)


def write_table(path, rows):
    """Write sparse measured columns; absent observations remain blank."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def disjoint_norm(norms, depth_only=False):
    """Combine norms only across disjoint parameter groups, never loss terms."""
    selected = [value for key, value in norms.items()
                if not depth_only or key.startswith("depth_")]
    if not selected or any(not finite(v) or v < 0 for v in selected):
        return None
    return math.sqrt(sum(v * v for v in selected))


def record_context(record, path, source):
    return {"arm": record.get("arm", path.parent.name),
            "split": record.get("split", "native_train_batch" if source == "train_steps" else "unspecified"),
            "mode": record.get("module_mode", record.get("mode", "train" if source == "train_steps" else "unspecified")),
            "step": record.get("step", 0), "source": source}


def aggregate_observations(records):
    """Average fixed-frame scalar observations, retaining observation counts."""
    groups = defaultdict(list)
    for context, values in records:
        groups[tuple(context.items())].append(values)
    result = []
    for context, values in sorted(groups.items()):
        row = dict(context)
        keys = sorted({key for value in values for key in value})
        row["observation_count"] = len(values)
        for key in keys:
            row[key] = avg(value.get(key) for value in values)
            row[key + " / observed"] = sum(finite(value.get(key)) for value in values)
        result.append(row)
    return result


def collect_training_and_audits(root):
    training, gradients, coverage, switches = [], [], [], []
    sources = defaultdict(list)
    for name in ("train_steps", "fixed_probe", "gradient_audit"):
        for path in sorted(root.rglob(name + ".jsonl")):
            sources[name].append(str(path.resolve()))
            records = read_jsonl(path)
            if name == "gradient_audit":
                # Depth DPT/FiLM/other groups are disjoint. Endpoint aliases are
                # excluded, because they are observations of the same graph.
                grouped = defaultdict(dict)
                duplicate_records = defaultdict(list)
                first_ordinals = defaultdict(dict)
                for ordinal, r in enumerate(records, 1):
                    if str(r.get("group", "")).startswith("depth_"):
                        key = (r.get("arm", path.parent.name), r.get("route", "unspecified"),
                               r.get("split", "unspecified"), r.get("step", 0),
                               r.get("batch", 0), tuple(r.get("indices") or ()),
                               r.get("scale", "raw"), r.get("loss", "unknown"))
                        group = r["group"]
                        if group in grouped[key]:
                            first = grouped[key][group]
                            # A repeated audit is not another disjoint parameter
                            # group. Preserve the first row and expose differences.
                            compared = ("norm", "state", "state_numel", "connected_numel",
                                        "total_numel", "loss_value", "weight")
                            differences = [field for field in compared if first.get(field) != r.get(field)]
                            duplicate_records[key].append({"group": group, "record_ordinal": ordinal,
                                                           "different_fields": differences})
                        else:
                            grouped[key][group] = r
                            first_ordinals[key][group] = ordinal
                for key, by_group in sorted(grouped.items()):
                    arm, route, split, step, batch, indices, scale, loss = key
                    components = list(by_group.values())
                    states = defaultdict(int)
                    for r in components:
                        for state, numel in r.get("state_numel", {}).items():
                            states[state] += int(numel)
                    present = [r["norm"] for r in components if finite(r.get("norm"))]
                    nonfinite = any(r.get("state") == "nonfinite" for r in components)
                    norm = math.sqrt(sum(v * v for v in present)) if present and not nonfinite else None
                    state = ("nonfinite" if nonfinite else "connected_nonzero" if norm and norm > 0
                             else "connected_zero" if norm == 0
                             else "not_requires_grad" if all(r.get("state") == "not_requires_grad" for r in components)
                             else "unused/None")
                    gradients.append({"arm": arm, "route": route, "split": split, "step": step,
                                      "batch": batch, "indices": json.dumps(indices), "scale": scale, "loss": loss,
                                      "depth_parameter_grad_norm": norm, "state": state,
                                      "connected_numel": sum(r.get("connected_numel", 0) for r in components),
                                      "state_numel": json.dumps(dict(states), sort_keys=True),
                                      "duplicate_records": len(duplicate_records[key]),
                                      "different_duplicate_records": sum(bool(r["different_fields"]) for r in duplicate_records[key]),
                                      "first_group_record_ordinals": json.dumps(first_ordinals[key], sort_keys=True),
                                      "duplicate_record_provenance": json.dumps(duplicate_records[key], sort_keys=True),
                                      "source_path": str(path.resolve())})
                continue
            for record in records:
                context = record_context(record, path, name)
                cov = {key: value for key, value in record.get("coverage", {}).items()
                       if key.startswith("D: Dynamics ") or key in {key for key, _ in COVERAGE_PANELS}}
                if cov:
                    coverage.append((context, cov))
                identity = record.get("identity_changes", {})
                if identity:
                    # Initial-reference / missing-identity rows are missing,
                    # never zero switch rates. They remain as gaps in plots.
                    switches.append((context, {key: identity.get(key) for key, _ in SWITCH_PANELS}))
                if name != "train_steps":
                    continue
                row = {**context, "seen_images": record.get("seen_images"),
                       "total_loss": record.get("loss"), "global_grad_preclip": record.get("global_grad_preclip"),
                       "clip_coefficient": record.get("clip_coefficient"),
                       "depth_grad_preclip": disjoint_norm(record.get("group_grad_preclip", {}), True),
                       "depth_grad_postclip": disjoint_norm(record.get("group_grad_postclip", {}), True),
                       "parameter_delta_norm": disjoint_norm(record.get("actual_parameter_update_norm", {})),
                       "depth_parameter_delta_norm": disjoint_norm(record.get("actual_parameter_update_norm", {}), True),
                       "source_path": str(path.resolve())}
                for scale in ("raw", "weighted"):
                    for loss in LOSS_TERMS:
                        row[scale + "_loss_" + loss] = record.get(scale + "_losses", {}).get(loss)
                training.append(row)
    return {"training": training, "gradients": gradients,
            "coverage": aggregate_observations(coverage),
            "switches": aggregate_observations(switches), "sources": dict(sources)}


def plot_panels(rows, panels, groups, path, title, symlog=False):
    """Plot measured samples only, leaving missing/unused observations as gaps."""
    if not rows or not any(finite(row.get(metric)) for row in rows for metric, _ in panels):
        return False
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
    ncols = min(3, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                               squeeze=False, constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, panels):
        lines = defaultdict(lambda: defaultdict(list))
        plotted_values = []
        for row in rows:
            lines[tuple(str(row.get(key, "unspecified")) for key in groups)][row["step"]].append(row.get(metric))
        for identity, by_step in sorted(lines.items()):
            points = [(step, avg(values)) for step, values in sorted(by_step.items())]
            if any(finite(value) for _, value in points):
                plotted_values.extend(value for _, value in points if finite(value))
                ax.plot([step for step, _ in points],
                        [value if finite(value) else math.nan for _, value in points],
                        marker=".", label=" / ".join(identity))
        ax.set(title=label, xlabel="Successful optimizer updates")
        ax.grid(alpha=.2)
        magnitudes = [abs(value) for value in plotted_values if value != 0]
        wide_range = bool(magnitudes) and max(magnitudes) / min(magnitudes) >= 20
        crosses_zero = bool(plotted_values) and min(plotted_values) < 0 < max(plotted_values)
        if symlog and metric != "clip_coefficient" and (wide_range or crosses_zero):
            ax.set_yscale("symlog", linthresh=max(min(magnitudes) * .5, 1e-12))
            # A symlog major locator can emit no in-range ticks for a narrow
            # visible interval. Guarantee readable numeric labels even then.
            low, high = ax.get_ylim()
            visible = [tick for tick in ax.get_yticks() if low <= tick <= high]
            if len(visible) < 2:
                ticks = sorted(set([min(plotted_values), max(plotted_values),
                                    *([0.0] if low <= 0 <= high else [])]))
                ax.yaxis.set_major_locator(FixedLocator(ticks))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:.3g}"))
        if ax.lines:
            ax.legend(fontsize=6)
        else:
            states = [row[metric + "__state"] for row in rows if metric + "__state" in row]
            message = ("No connected gradients (None)" if states and all(state == "unused/None" for state in states)
                       else "Parameters do not require gradients" if states and all(state == "not_requires_grad" for state in states)
                       else "Nonfinite gradient measurements" if "nonfinite" in states
                       else "No finite measurements")
            ax.text(.5, .5, message, ha="center", transform=ax.transAxes, fontsize=9)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    figure.suptitle(title, fontsize=11)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def render_training_and_audits(data, figures):
    generated = []

    def draw(rows, panels, groups, filename, title, symlog=False):
        if plot_panels(rows, panels, groups, figures / filename, title, symlog):
            generated.append(filename)

    for scale in ("raw", "weighted"):
        draw(data["training"], [(scale + "_loss_" + loss, loss.capitalize()) for loss in LOSS_TERMS],
             ("arm",), f"training_losses_{scale}.png",
             f"Native training batches: {scale} loss terms (changing batch/target composition)")
    draw(data["training"], [("global_grad_preclip", "Global gradient norm before clipping"),
         ("clip_coefficient", "Global clipping coefficient"),
         ("parameter_delta_norm", "Actual parameter update norm (all groups)"),
         ("depth_grad_preclip", "Depth-parameter total gradient before clipping"),
         ("depth_grad_postclip", "Depth-parameter total gradient after clipping"),
         ("depth_parameter_delta_norm", "Actual depth-parameter update norm")],
         ("arm",), "training_optimization.png",
         "Measured optimization; clipping coefficient is not an Adam learning-rate multiplier", True)
    for scale in ("raw", "weighted"):
        by_measurement = defaultdict(dict)
        for row in data["gradients"]:
            if row["scale"] != scale:
                continue
            key = tuple(row[k] for k in ("arm", "route", "split", "step", "batch", "indices", "source_path"))
            by_measurement[key][row["loss"]] = row["depth_parameter_grad_norm"]
            by_measurement[key][row["loss"] + "__state"] = row["state"]
        rows = [{**dict(zip(("arm", "route", "split", "step", "batch", "indices", "source_path"), key)), **values}
                for key, values in by_measurement.items()]
        draw(rows, [(loss, loss.capitalize() + " loss → depth-parameter gradient norm") for loss in LOSS_TERMS],
             ("arm", "route", "split"), f"depth_parameter_gradients_{scale}.png",
             f"{scale.capitalize()} per-loss gradients; means across audit batches; unused paths omitted", True)
    for source, filename, title in (("fixed_probe", "fixed_probe_coverage.png", "Mean per fixed-frame probe; native labels"),
                                     ("train_steps", "training_coverage.png", "Native training-batch coverage; batch composition varies")):
        draw([row for row in data["coverage"] if row["source"] == source], COVERAGE_PANELS,
             ("arm", "split", "mode"), filename, title)
    draw(data["switches"], SWITCH_PANELS, ("arm", "split", "mode"), "fixed_probe_identity_changes.png",
         "Changes since the previous probe of the same frame/mode; aligned-slot and same-identity rates differ")
    return generated


def render_local_geometry(measured, output):
    """Retain signed local geometry on common axes, with a factored legend."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    arms = sorted({arm for arm, _, _ in measured})
    splits = sorted({split for _, split, _ in measured})
    modes = sorted({mode for _, _, mode in measured})
    palette = plt.get_cmap("tab10")
    known_colors = {"P0": "#777777", "D0": "#0072B2", "D1": "#D55E00"}
    colors = {arm: known_colors.get(arm, palette(index % 10)) for index, arm in enumerate(arms)}
    line_options = ("-", "--", "-.", ":")
    split_styles = {split: line_options[index % len(line_options)] for index, split in enumerate(splits)}
    if "train" in split_styles:
        split_styles["train"] = "-"
        for index, split in enumerate(value for value in splits if value != "train"):
            split_styles[split] = line_options[1 + index % (len(line_options) - 1)]
    mode_markers = {mode: {"train": "^", "eval": "o"}.get(mode, ("s", "D", "v")[index % 3])
                    for index, mode in enumerate(modes)}
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = (("local_slope", "Local depth-difference slope", "Slope (dimensionless)"),
              ("local_correlation", "Local depth-difference correlation", "Pearson correlation"),
              ("foreground_mae_m", "Foreground depth MAE", "MAE (m)"),
              ("foreground_bias_m", "Foreground depth bias", "Prediction minus GT (m)"))
    for ax, (metric, title, ylabel) in zip(axes.flat, panels):
        ax.axhline(0., color="#777777", linewidth=.8, linestyle=":", zorder=0)
        if metric in ("local_slope", "local_correlation"):
            ax.axhline(1., color="#aaaaaa", linewidth=.8, linestyle=":", zorder=0)
        has_values = False
        for (arm, split, mode), trajectory in sorted(measured.items()):
            points = [(row["step"], row.get(metric)) for row in trajectory]
            if any(finite(value) for _, value in points):
                has_values = True
                ax.plot([step for step, _ in points],
                        [value if finite(value) else math.nan for _, value in points],
                        color=colors[arm], linestyle=split_styles[split], marker=mode_markers[mode],
                        markersize=4, linewidth=1.25, alpha=.85)
        ax.set(title=title, xlabel="Successful optimizer updates", ylabel=ylabel)
        ax.grid(alpha=.15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        if metric == "local_correlation":
            ax.set_ylim(-1.05, 1.05)
        elif metric == "foreground_mae_m":
            ax.set_ylim(bottom=0.)
        if not has_values:
            ax.text(.5, .5, "No finite measurements", ha="center", transform=ax.transAxes)
    handles = [Line2D([], [], color=colors[arm], linewidth=2, label=f"Arm: {arm}") for arm in arms]
    handles += [Line2D([], [], color="#444444", linestyle=split_styles[split], label=f"Split: {split}")
                for split in splits]
    handles += [Line2D([], [], color="#444444", linestyle="none", marker=mode_markers[mode],
                       markersize=5, label=f"Mode: {mode}") for mode in modes]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    figure.suptitle("Fixed-frame local geometry and foreground depth\n"
                   "Imagewise means; fixed same-instance pairs; dotted references at 0 and identity slope/correlation 1",
                   fontsize=11, y=.98)
    figure.subplots_adjust(left=.09, right=.98, top=.86, bottom=.20, wspace=.30, hspace=.47)
    figure.savefig(output / "local_geometry.png", dpi=160)
    plt.close(figure)


def render_fixed_frames(root, output):
    """One fixed frame, shared metric colour scales across arms and times."""
    files = sorted(root.rglob("fixed_frame_*.pt"))
    if not files:
        return
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    by_step = defaultdict(dict)
    for path in files:
        state = torch.load(path, map_location="cpu", weights_only=False)
        by_step[int(state["step"])][path.parent.name] = state
    for step, states in sorted(by_step.items()):
        arms = [arm for arm in ("D0", "D1") if arm in states]
        arms += sorted(set(states) - set(arms))
        if not arms:
            continue
        first = states[arms[0]]
        rgb = first["rgb"][0].float().numpy().transpose(1, 2, 0)
        rgb = np.clip(rgb * np.array([.229, .224, .225]) + np.array([.485, .456, .406]), 0, 1)
        gt = first["gt"].squeeze().float().numpy()
        figure, axes = plt.subplots(1, 2 + 2 * len(arms), figsize=(4 * (2 + 2 * len(arms)), 4), constrained_layout=True)
        axes[0].imshow(rgb)
        axes[0].set_title("Fixed RGB frame")
        depth_artist = axes[1].imshow(gt, cmap="viridis", norm=Normalize(0, 1))
        axes[1].set_title("GT depth (m)")
        error_artist = None
        for i, arm in enumerate(arms):
            value = states[arm]
            if not torch.equal(first["gt"], value["gt"]) or not torch.equal(first["rgb"], value["rgb"]):
                raise ValueError(f"Cannot pair different frames at update {step}: {arm}")
            pred = value["pred"].squeeze().float().numpy()
            axes[2 + i * 2].imshow(pred, cmap="viridis", norm=Normalize(0, 1))
            axes[2 + i * 2].set_title(f"{arm} depth (m)")
            error = np.where((gt >= .2) & (gt <= 1), pred - gt, np.nan)
            error_artist = axes[3 + i * 2].imshow(error, cmap="coolwarm", norm=Normalize(-.1, .1))
            axes[3 + i * 2].set_title(f"{arm} error (m)")
        for axis in axes:
            axis.axis("off")
        figure.colorbar(depth_artist, ax=[axes[1], *[axes[2 + 2*i] for i in range(len(arms))]], shrink=.65)
        figure.colorbar(error_artist, ax=[axes[3 + 2*i] for i in range(len(arms))], shrink=.65)
        figure.suptitle(f"Successful optimizer updates: {step}; shared metre scales")
        figure.savefig(output / f"paired_depth_step_{step:06d}.png", dpi=140)
        plt.close(figure)


def render_gradient_maps(files, output):
    """Render saved output gradients with one locked signed scale per endpoint.

    Scales cover EVERY loaded route and both weighted objectives. None and
    connected-zero maps retain different visual/metadata states. This function
    reads local .pt snapshots and writes only PNGs and compact JSON metadata.
    """
    if not files:
        return []
    import hashlib
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, SymLogNorm
    from matplotlib.ticker import FuncFormatter

    endpoints = (("depth_net_pred", "metric_depth", "metric depth z (m)"),
                 ("depth_head_raw_pred", "raw_depth", "raw depth output (dimensionless)"))
    objectives = ("depth", "task")
    bundles = defaultdict(dict)
    loaded = []
    for path in files:
        state = torch.load(path, map_location="cpu", weights_only=False)
        key = (str(path.parent.resolve()), int(state["step"]), int(state["batch_id"]))
        route = str(state["route"])
        if route in bundles[key]:
            raise ValueError(f"Duplicate gradient snapshot for {key}, route={route}")
        item = {"path": str(path.resolve()), "state": state, "maps": {}, "statistics": {}}
        for endpoint, _, _ in endpoints:
            for objective in objectives:
                name = objective + "_wrt_" + endpoint
                if name not in state["gradients"]:
                    raise ValueError(f"Missing gradient snapshot key {name}: {path}")
                tensor = state["gradients"][name]
                if tensor is None:
                    item["maps"][name] = None
                    item["statistics"][name] = {"state": "disconnected", "tensor_is_none": True}
                    continue
                value = tensor.detach().squeeze().float().numpy()
                if value.ndim != 2:
                    raise ValueError(f"Gradient must reduce to one HxW map: {path}, {name}, {value.shape}")
                good = np.isfinite(value)
                finite_values = value[good]
                nonzero = int(np.count_nonzero(finite_values))
                item["maps"][name] = value
                item["statistics"][name] = {
                    "state": "nonfinite" if not bool(good.all()) else "connected_nonzero" if nonzero else "connected_zero",
                    "tensor_is_none": False, "shape": list(value.shape),
                    "finite_count": int(good.sum()), "nonfinite_count": int((~good).sum()),
                    "nonzero_finite_count": nonzero, "total_count": int(value.size),
                    "max_abs_finite": float(np.abs(finite_values).max()) if finite_values.size else None,
                    "l2_norm_finite": float(np.linalg.norm(finite_values.astype(np.float64))),
                }
        bundles[key][route] = item
        loaded.append(item)

    metadata = {
        "scope": "Image 0 of saved initial audit batch; local output-space gradients, not actual network/optimizer updates",
        "objectives": {"depth": "actual weighted metric-depth loss", "task": "sum of actual weighted non-depth losses"},
        "normalization_scope": "one global signed scale per observation point across every loaded route, objective and bundle",
        "metric_depth_context_range_m": [0.0, 1.0],
        "rgb_display": "ImageNet mean/std inverse normalization, then clipping to [0,1]",
        "sources": [item["path"] for item in loaded], "scales": {}, "artifacts": [],
        "none_semantics": "disconnected: saved gradient was None; distinct from a connected tensor containing zeros",
        "nonfinite_display": "masked pixels are gray; finite color scale excludes NaN/Inf and counts are reported",
    }
    norms = {}
    for endpoint, _, _ in endpoints:
        maxima = [item["statistics"][objective + "_wrt_" + endpoint].get("max_abs_finite")
                  for item in loaded for objective in objectives]
        maxima = [value for value in maxima if finite(value)]
        observed = max(maxima) if maxima else None
        bound = observed if observed is not None and observed > 0 else 1.0
        # Pool every finite nonzero pixel before choosing a SINGLE endpoint
        # threshold. A sparse large task derivative must not wash out the dense
        # depth-loss maps; the signed maximum remains exact and unclipped.
        nonzero_chunks = []
        for item in loaded:
            for objective in objectives:
                value = item["maps"][objective + "_wrt_" + endpoint]
                if value is not None:
                    magnitudes = np.abs(value[np.isfinite(value) & (value != 0)])
                    if magnitudes.size:
                        nonzero_chunks.append(magnitudes)
        pooled_count = sum(int(chunk.size) for chunk in nonzero_chunks)
        pooled_median = (float(np.median(np.concatenate(nonzero_chunks).astype(np.float64)))
                         if nonzero_chunks else None)
        maximum_based_threshold = bound * 1e-3
        threshold = max(min(maximum_based_threshold, pooled_median)
                        if pooled_median is not None else maximum_based_threshold, 1e-300)
        norms[endpoint] = SymLogNorm(linthresh=threshold, linscale=1.0, vmin=-bound, vmax=bound, base=10)
        metadata["scales"][endpoint] = {
            "normalization": "SymLogNorm", "colormap": "RdBu_r", "base": 10,
            "observed_global_max_abs_finite": observed, "vmin": -bound, "vmax": bound,
            "linthresh": threshold, "linthresh_fraction_of_global_display_max": threshold / bound,
            "linthresh_formula": "max(min(global_display_max_abs * 1e-3, pooled_nonzero_abs_median), 1e-300); without nonzero pixels use global_display_max_abs * 1e-3",
            "linthresh_maximum_based_candidate": maximum_based_threshold,
            "pooled_nonzero_abs_median": pooled_median, "pooled_nonzero_finite_count": pooled_count,
            "pooled_median_scope": "all finite nonzero pixels, equally weighted, across every loaded route, objective and bundle for this endpoint",
            "linscale": 1.0,
            "zero_or_no_finite_data_display_fallback": observed is None or observed == 0,
            "locked_before_rendering": True,
        }
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#9e9e9e")
    paths = []
    preferred = ("none", "gse", "seed_xyz", "support", "all")
    for (directory, step, batch_id), items in sorted(bundles.items()):
        routes = [route for route in preferred if route in items] + sorted(set(items) - set(preferred))
        first = items[routes[0]]["state"]
        for route, item in items.items():
            state = item["state"]
            for name in ("rgb", "gt_m"):
                if not torch.equal(first[name], state[name]):
                    raise ValueError(f"Gradient route panels have different frame inputs: {directory}, {route}, {name}")
            for name in ("pred_m", "raw"):
                if not torch.allclose(first[name], state[name], atol=1e-6, rtol=1e-5):
                    raise ValueError(f"Gradient route panels have different forward values: {directory}, {route}, {name}")
        rgb = first["rgb"].float().numpy().transpose(1, 2, 0)
        rgb = np.clip(rgb * np.array([.229, .224, .225]) + np.array([.485, .456, .406]), 0, 1)
        gt = first["gt_m"].squeeze().float().numpy()
        pred = first["pred_m"].squeeze().float().numpy()
        suffix = "" if len(bundles) == 1 else "_" + hashlib.sha256(directory.encode()).hexdigest()[:8] + f"_s{step}_b{batch_id}"
        for endpoint, slug, endpoint_label in endpoints:
            columns = max(len(routes), 3)
            figure, axes = plt.subplots(3, columns, figsize=(3.4 * columns, 10.0),
                                       squeeze=False, constrained_layout=True)
            axes[0, 0].imshow(rgb)
            axes[0, 0].set_title("Same audited RGB frame")
            depth_artist = axes[0, 1].imshow(gt, cmap="viridis", norm=Normalize(0, 1))
            axes[0, 1].set_title("GT depth (m)")
            axes[0, 2].imshow(pred, cmap="viridis", norm=Normalize(0, 1))
            axes[0, 2].set_title("Predicted depth (m)")
            for ax in axes[0]:
                ax.axis("off")
            figure.colorbar(depth_artist, ax=[axes[0, 1], axes[0, 2]], shrink=.7,
                            label="Depth: fixed 0–1 m scale")
            scale = metadata["scales"][endpoint]
            panels = []
            for row, objective in enumerate(objectives, 1):
                for column, route in enumerate(routes):
                    ax = axes[row, column]
                    name = objective + "_wrt_" + endpoint
                    value = items[route]["maps"][name]
                    statistics = items[route]["statistics"][name]
                    ax.set_title(f"{route} · weighted {objective}", fontsize=10)
                    if value is None:
                        ax.set_facecolor("#eeeeee")
                        ax.text(.5, .5, "Disconnected\n(gradient is None)", ha="center", va="center", transform=ax.transAxes)
                    else:
                        ax.imshow(np.ma.masked_invalid(value), cmap=cmap, norm=norms[endpoint], interpolation="nearest")
                        if statistics["state"] == "connected_zero":
                            ax.text(.5, .5, "Connected zero", ha="center", va="center", color="#333333", transform=ax.transAxes)
                        elif statistics["state"] == "nonfinite":
                            ax.text(.02, .02, f"Nonfinite: {statistics['nonfinite_count']} pixels", color="black",
                                    transform=ax.transAxes, bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
                    ax.set_xticks([])
                    ax.set_yticks([])
                    panels.append({"route": route, "objective": objective, **statistics})
                for ax in axes[row, len(routes):]:
                    ax.axis("off")
            bar = figure.colorbar(ScalarMappable(norm=norms[endpoint], cmap=cmap), ax=list(axes[1:].flat),
                                 shrink=.85, label=f"Signed derivative w.r.t. {endpoint_label}; shared across every route/objective")
            bound, threshold = scale["vmax"], scale["linthresh"]
            bar.set_ticks([-bound, -threshold, 0.0, threshold, bound])
            bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:.3g}"))
            figure.suptitle(f"Local gradients w.r.t. {endpoint_label} · update {step}, audit batch {batch_id}, image 0\n"
                           f"One signed SymLog scale: ±{bound:.3g}, linear threshold {threshold:.3g}; not parameter updates", fontsize=12)
            path = output / f"gradient_spatial_{slug}{suffix}.png"
            figure.savefig(path, dpi=160)
            plt.close(figure)
            paths.append(path)
            metadata["artifacts"].append({"png_path": str(path.resolve()), "observation_point": endpoint,
                                           "source_directory": directory, "step": step, "batch_id": batch_id,
                                           "image_index": 0, "routes": routes,
                                           "missing_standard_routes": [route for route in preferred if route not in items],
                                           "panels": panels})
    metadata_path = output.parent / "gradient_spatial_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths


def summarize(root, output, plots=True):
    output.mkdir(parents=True, exist_ok=True)
    auxiliary = collect_training_and_audits(root)
    gradient_map_files = sorted(root.rglob("gradient_maps_*_initial.pt"))
    for name, values in (("training_summary", auxiliary["training"]),
                         ("gradient_summary", auxiliary["gradients"]),
                         ("coverage_summary", auxiliary["coverage"]),
                         ("identity_changes_summary", auxiliary["switches"])):
        write_table(output / (name + ".csv"), values)
    grouped = defaultdict(list)
    sources = []
    per_image = []
    for path in sorted(root.rglob("fixed_probe.jsonl")):
        sources.append(str(path.resolve()))
        for record in read_jsonl(path):
            step = record.get("step", record.get("optimizer_step", record.get("update", 0)))
            arm = record.get("arm", path.parent.name)
            split = record.get("split", "unspecified")
            mode = record.get("mode", record.get("module_mode", record.get("measurement_mode", "unspecified")))
            metrics = flatten_metrics(record)
            grouped[(arm, split, mode, step)].extend(metrics)
            for metric in metrics:
                image_row = {"arm": arm, "split": split, "mode": mode, "step": step,
                             "index": record.get("index"), "scene": record.get("scene"), "frame": record.get("frame")}
                for region, values in metric.get("regions", {}).items():
                    for name, value in values.items():
                        if not isinstance(value, (dict, list)):
                            image_row[f"{region}_{name}"] = value
                        elif isinstance(value, dict):
                            image_row.update({f"{region}_{name}_{k}": v for k, v in value.items()})
                for scope in ("local", "raw"):
                    image_row.update({f"{scope}_{k}": v for k, v in metric.get(scope, {}).items()
                                      if not isinstance(v, (dict, list))})
                per_image.append(image_row)
    write_table(output / "per_image.csv", per_image)
    events = {str(path.resolve()): read_jsonl(path) for path in sorted(root.rglob("events.jsonl"))}
    (output / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    rows = []
    for (arm, split, mode, step), images in sorted(grouped.items()):
        valid = [im.get("regions", {}).get("valid", {}) for im in images]
        local = [im.get("local", {}) for im in images]
        foreground = [im.get("regions", {}).get("foreground", {}) for im in images]
        eligible = [x for x in foreground if x.get("count", 0) >= 2 and finite(x.get("gt_std")) and x["gt_std"] > .005]
        image_means = [x["pred_mean"] for x in valid if finite(x.get("pred_mean"))]
        foreground_means = [x["pred_mean"] for x in foreground if finite(x.get("pred_mean"))]
        flat = [x for x in eligible if finite(x.get("std_ratio")) and x["std_ratio"] < .1]
        rows.append({"arm": arm, "split": split, "mode": mode, "step": step,
                     "image_count": len(images), "eligible_image_count": len(eligible),
                     "mae_m": avg(x.get("mae") for x in valid),
                     "bias_m": avg(x.get("bias") for x in valid),
                     "mean_image_std_ratio": avg(x.get("std_ratio") for x in valid),
                     "between_image_pred_mean_population_std_m": pstdev(image_means) if image_means else None,
                     "between_image_foreground_mean_population_std_m": pstdev(foreground_means) if foreground_means else None,
                     "below_min_fraction": avg(x.get("below_min_fraction") for x in valid),
                     "foreground_mae_m": avg(x.get("mae") for x in foreground),
                     "foreground_bias_m": avg(x.get("bias") for x in foreground),
                     "foreground_pred_std_m": avg(x.get("pred_std") for x in foreground),
                     "foreground_gt_std_m": avg(x.get("gt_std") for x in foreground),
                     "foreground_mean_image_std_ratio": avg(x.get("std_ratio") for x in foreground),
                     "flat_image_fraction": len(flat) / len(eligible) if eligible else None,
                     "local_contrast_ratio": avg(x.get("contrast_ratio") for x in local),
                     "local_slope": avg(x.get("slope") for x in local),
                     "local_correlation": avg(x.get("correlation") for x in local),
                     "local_difference_mae_m": avg(x.get("difference_mae") for x in local),
                     "sigmoid_derivative_mean": avg(im.get("raw", {}).get("sigmoid_derivative_mean") for im in images)})
    csv_path = output / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        else:
            handle.write("arm,split,mode,step,image_count\n")

    measured = defaultdict(list)
    for row in rows:
        measured[(row["arm"], row["split"], row["mode"])].append(row)
    report = ["# CVA metric-depth dynamics: measured results", "",
              "These tables summarize fixed-frame depth, training, gradient and coverage measurements. They do not establish "
              "a necessary/sufficient route or a causal mechanism.", ""]
    if not rows:
        report.append("No fixed-probe measurements are available; no training conclusion is supported.")
    else:
        report += ["| Arm | Split / mode | Last update | MAE (m) | Mean image std ratio | Flat image fraction |",
                   "| --- | --- | ---: | ---: | ---: | ---: |"]
        fmt = lambda x: f"{x:.6g}" if finite(x) else "missing"
        for (arm, split, mode), trajectory in sorted(measured.items()):
            last = trajectory[-1]
            report.append(f"| {arm} | {split} / {mode} | {last['step']} | {fmt(last['mae_m'])} | "
                          f"{fmt(last['mean_image_std_ratio'])} | {fmt(last['flat_image_fraction'])} |")
    contracts = []
    for path in sorted(root.rglob("contract.json")):
        contract = json.loads(path.read_text(encoding="utf-8"))
        contracts.append({"path": str(path.resolve()), "contract": contract})
    report += ["", "## Scope and unresolved questions", "",
               f"- Available records: {len(auxiliary['training'])} training steps; "
               f"{len(auxiliary['gradients'])} per-loss depth-parameter gradient observations; "
               f"{len(auxiliary['coverage'])} coverage summaries; {len(auxiliary['switches'])} probe identity summaries.",
               "- Loss plots retain raw and actual weighted units separately; native batches and targets can change.",
               "- Gradient norms combine disjoint depth-parameter groups in quadrature, separately for each loss. "
               "Task-loss norms are never added to estimate a total task gradient. Disconnected gradients remain missing.",
               f"- Repeated gradient audit records: {sum(r['duplicate_records'] for r in auxiliary['gradients'])} "
               f"duplicates, including {sum(r['different_duplicate_records'] for r in auxiliary['gradients'])} with "
               "different measured fields. For the same source/arm/route/split/update/batch/indices/scale/loss/group, "
               "the first record is retained; repetitions are neither summed nor silently averaged. "
               "Original record ordinals and differing fields are recorded in gradient_summary.csv.",
               "- Actual parameter deltas are measured optimizer updates. Global clipping coefficients are not Adam learning-rate multipliers.",
               f"- Saved spatial gradient snapshots: {len(gradient_map_files)}. When plotted, signed scales are fixed globally "
               "across routes and depth/task objectives separately for metric/raw observation points. "
               "None is labeled disconnected, not zero. These maps show local output derivatives, not optimizer updates. "
               "Artifact paths and scale parameters are recorded in gradient_spatial_metadata.json.",
               "- Probe coverage is averaged per fixed frame. Identity changes compare the same frame/mode to its previous probe; "
               "aligned-slot changes can include changed queries, whereas same-identity NN changes condition on matching query IDs.",
               "- Depth summaries are means of finite per-image metrics. Existing MAE/bias/std-ratio columns retain their "
               "GT-valid-region meaning; foreground_* columns use the foreground region. Predicted/GT standard deviations "
               "and MAE/bias are in metres. Local slope, correlation and difference MAE use the saved fixed same-instance pairs; "
               "signed slope/correlation reveal inversions that contrast magnitude alone cannot identify. "
               "local_geometry.png uses common axes across arms/splits/modes, with no per-frame normalization.",
               "- Reproduction: interpret only the measured initialization, route, data protocol and update budget.",
               "- Necessary/sufficient path: not established by these summary statistics.",
               "- Mechanism: requires a matched failure plus pre-failure actual-update interventions.",
               "- Exclusions: no absent mechanism is ruled out by a short stable trajectory.",
               "- Validation: test_seen is user-designated validation; its original split identity remains in the contract.",
               "- Next step: inspect audit connectivity and paired curves before extending the update budget or opening P3.",
               "", "Source contracts and measured files are indexed in `summary_sources.json`."]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "summary_sources.json").write_text(json.dumps({"fixed_probe_files": sources,
        "gradient_snapshot_files": [str(path.resolve()) for path in gradient_map_files],
        "additional_measured_files": auxiliary["sources"], "contracts": contracts}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    generated = []
    if plots and (rows or gradient_map_files or any(auxiliary[key] for key in ("training", "gradients", "coverage", "switches"))):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figures = output / "figures"
        figures.mkdir(exist_ok=True)
        generated.extend(render_training_and_audits(auxiliary, figures))
        generated.extend(path.name for path in render_gradient_maps(gradient_map_files, figures))
    if plots and rows:
        figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        metrics = [("mae_m", "Depth MAE (m)"), ("bias_m", "Depth bias (m)"),
                   ("mean_image_std_ratio", "Mean within-image std ratio"),
                   ("local_contrast_ratio", "Local contrast ratio"),
                   ("below_min_fraction", "Depth below minimum"),
                   ("sigmoid_derivative_mean", "Mean sigmoid derivative")]
        for ax, (metric, title) in zip(axes.flat, metrics):
            for (arm, split, mode), trajectory in sorted(measured.items()):
                points = [(r["step"], r[metric]) for r in trajectory if finite(r[metric])]
                if points:
                    ax.plot(*zip(*points), marker=".", label=f"{arm}/{split}/{mode}")
            ax.set(title=title, xlabel="Successful optimizer updates")
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=7)
        figure.savefig(figures / "depth_dynamics.png", dpi=160)
        plt.close(figure)
        generated.append("depth_dynamics.png")
        render_local_geometry(measured, figures)
        generated.append("local_geometry.png")
        render_fixed_frames(root, figures)
    return {"rows": len(rows), "training_rows": len(auxiliary["training"]),
            "gradient_rows": len(auxiliary["gradients"]), "figures": generated,
            "gradient_spatial_metadata": str(output / "gradient_spatial_metadata.json") if plots and gradient_map_files else None,
            "summary": str(csv_path), "report": str(output / "report.md")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Run diagnostics root")
    parser.add_argument("--output", required=True, type=Path, help="Summary directory")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(summarize(args.input, args.output, not args.no_plots), indent=2))


if __name__ == "__main__":
    main()
