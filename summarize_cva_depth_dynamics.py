#!/usr/bin/env python3
"""Summarize measured CVA depth trajectories without inferring a mechanism."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


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
    ncols = min(3, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                               squeeze=False, constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, panels):
        lines = defaultdict(lambda: defaultdict(list))
        for row in rows:
            lines[tuple(str(row.get(key, "unspecified")) for key in groups)][row["step"]].append(row.get(metric))
        for identity, by_step in sorted(lines.items()):
            points = [(step, avg(values)) for step, values in sorted(by_step.items())]
            if any(finite(value) for _, value in points):
                ax.plot([step for step, _ in points],
                        [value if finite(value) else math.nan for _, value in points],
                        marker=".", label=" / ".join(identity))
        ax.set(title=label, xlabel="Successful optimizer updates")
        ax.grid(alpha=.2)
        if symlog and metric != "clip_coefficient":
            ax.set_yscale("symlog", linthresh=1e-8)
        if ax.lines:
            ax.legend(fontsize=6)
        else:
            ax.text(.5, .5, "No finite measurements", ha="center", transform=ax.transAxes, fontsize=9)
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


def summarize(root, output, plots=True):
    output.mkdir(parents=True, exist_ok=True)
    auxiliary = collect_training_and_audits(root)
    for name, values in (("training_summary", auxiliary["training"]),
                         ("gradient_summary", auxiliary["gradients"]),
                         ("coverage_summary", auxiliary["coverage"]),
                         ("identity_changes_summary", auxiliary["switches"])):
        write_table(output / (name + ".csv"), values)
    grouped = defaultdict(list)
    sources = []
    for path in sorted(root.rglob("fixed_probe.jsonl")):
        sources.append(str(path.resolve()))
        for record in read_jsonl(path):
            step = record.get("step", record.get("optimizer_step", record.get("update", 0)))
            arm = record.get("arm", path.parent.name)
            split = record.get("split", "unspecified")
            mode = record.get("mode", record.get("module_mode", record.get("measurement_mode", "unspecified")))
            grouped[(arm, split, mode, step)].extend(flatten_metrics(record))
    rows = []
    for (arm, split, mode, step), images in sorted(grouped.items()):
        valid = [im.get("regions", {}).get("valid", {}) for im in images]
        local = [im.get("local", {}) for im in images]
        foreground = [im.get("regions", {}).get("foreground", {}) for im in images]
        eligible = [x for x in foreground if finite(x.get("gt_std")) and x["gt_std"] > .005]
        flat = [x for x in eligible if finite(x.get("std_ratio")) and x["std_ratio"] < .1]
        rows.append({"arm": arm, "split": split, "mode": mode, "step": step,
                     "image_count": len(images), "eligible_image_count": len(eligible),
                     "mae_m": avg(x.get("mae") for x in valid),
                     "bias_m": avg(x.get("bias") for x in valid),
                     "mean_image_std_ratio": avg(x.get("std_ratio") for x in valid),
                     "below_min_fraction": avg(x.get("below_min_fraction") for x in valid),
                     "flat_image_fraction": len(flat) / len(eligible) if eligible else None,
                     "local_contrast_ratio": avg(x.get("contrast_ratio") for x in local),
                     "local_slope": avg(x.get("slope") for x in local),
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
               "- Probe coverage is averaged per fixed frame. Identity changes compare the same frame/mode to its previous probe; "
               "aligned-slot changes can include changed queries, whereas same-identity NN changes condition on matching query IDs.",
               "- Reproduction: interpret only the measured initialization, route, data protocol and update budget.",
               "- Necessary/sufficient path: not established by these summary statistics.",
               "- Mechanism: requires a matched failure plus pre-failure actual-update interventions.",
               "- Exclusions: no absent mechanism is ruled out by a short stable trajectory.",
               "- Validation: test_seen is user-designated validation; its original split identity remains in the contract.",
               "- Next step: inspect audit connectivity and paired curves before extending the update budget or opening P3.",
               "", "Source contracts and measured files are indexed in `summary_sources.json`."]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "summary_sources.json").write_text(json.dumps({"fixed_probe_files": sources,
        "additional_measured_files": auxiliary["sources"], "contracts": contracts}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    generated = []
    if plots and (rows or any(auxiliary[key] for key in ("training", "gradients", "coverage", "switches"))):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figures = output / "figures"
        figures.mkdir(exist_ok=True)
        generated.extend(render_training_and_audits(auxiliary, figures))
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
        render_fixed_frames(root, figures)
    return {"rows": len(rows), "training_rows": len(auxiliary["training"]),
            "gradient_rows": len(auxiliary["gradients"]), "figures": generated,
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
