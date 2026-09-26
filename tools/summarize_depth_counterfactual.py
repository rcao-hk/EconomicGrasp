"""Export validated P4 reports to imagewise and aggregate intervention CSVs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


CHECKS = ("normal_repeat_exact_state_match", "identical_pre_update_state_all_branches",
          "fixed_clip_non_depth_updates_exactly_match_normal", "source_state_unmodified",
          "restored_checkpoint_model_exact", "restored_checkpoint_optimizer_exact",
          "restored_checkpoint_rng_exact", "restored_loader_exact", "restored_runtime_exact",
          "restored_routes_exact", "cleared_parameter_gradients")


def metrics(row):
    value = row["metrics"]
    if len(value) != 1:
        raise ValueError("Expected the original one-image fixed probes")
    value = value[0]
    result = {f"{region}_{key}_m": value["regions"][region][key]
              for region in ("valid", "foreground", "background") for key in ("mae", "bias")}
    result.update({f"local_{key}": value["local"][key]
                   for key in ("slope", "correlation", "contrast_ratio", "difference_mae")})
    result["sigmoid_derivative_mean"] = value["raw"]["sigmoid_derivative_mean"]
    result["sigmoid_extreme_fraction"] = value["raw"]["sigmoid_extreme_fraction"]
    return result


def identity(row):
    return row["split"], row["module_mode"], row["index"]


def mean(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.mean(values) if values else None


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new summary directory: {output}")
    images, aggregates, sources, seen = [], [], [], set()
    for filename in args.reports:
        path = Path(filename).resolve()
        payload = path.read_bytes()
        report = json.loads(payload)
        if report["status"] != "completed" or not all(report.get(key) is True for key in CHECKS):
            raise ValueError(f"Incomplete or invalid replay controls: {path}")
        step, target = report["source_checkpoint"]["step"], report["target_loss"]
        if (step, target) in seen:
            raise ValueError("Duplicate source step/target; summarize independent runs separately")
        seen.add((step, target))
        before = {identity(row): metrics(row) for row in report["after_forward_pre_optimizer_probe"]}
        normal = {identity(row): metrics(row) for row in report["branches"]["normal_A"]["after_update_probe"]}
        if before.keys() != normal.keys():
            raise ValueError("Reference probe identities differ")
        for branch, data in report["branches"].items():
            rows = data["after_update_probe"]
            if {identity(row) for row in rows} != before.keys() or len(rows) != len(before):
                raise ValueError("Branch has changed or duplicate probe identities")
            grouped = {}
            for row in rows:
                key = identity(row)
                values = metrics(row)
                result = dict(source_step=step, target_loss=target, branch=branch,
                              split=key[0], mode=key[1], index=key[2], scene=row["scene"], frame=row["frame"])
                result.update(values)
                for name, reference in (("before_optimizer", before[key]), ("normal_A", normal[key])):
                    result.update({f"delta_{metric}_from_{name}": value - reference[metric]
                                   if value is not None and reference[metric] is not None else None
                                   for metric, value in values.items()})
                images.append(result)
                grouped.setdefault(key[:2], []).append(result)
            scope = set(report["scope_groups"])
            delta = data["parameter_delta_from_checkpoint"]["groups"]
            for (split, mode), group in grouped.items():
                result = dict(source_step=step, target_loss=target, branch=branch,
                              split=split, mode=mode, images=len(group),
                              global_grad_preclip=data["global_grad_preclip"],
                              clip_coefficient=data["clip_coefficient"],
                              depth_parameter_update_l2=math.sqrt(sum(v["l2"] ** 2 for k, v in delta.items() if k in scope)))
                metric_names = [key for key in group[0] if key not in
                                ("source_step", "target_loss", "branch", "split", "mode", "index", "scene", "frame")]
                result.update({key: mean(row[key] for row in group) for key in metric_names})
                result["images_with_lower_mae_than_normal_A"] = sum(
                    row["delta_valid_mae_m_from_normal_A"] < 0 for row in group
                    if row["delta_valid_mae_m_from_normal_A"] is not None)
                aggregates.append(result)
        sources.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
                        "source_checkpoint": report["source_checkpoint"], "git": report["git"],
                        "executed_source_sha256": report.get("executed_source_sha256"),
                        "controls": {key: report[key] for key in CHECKS}})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "interventions.csv", aggregates)
    write_csv(output / "interventions_per_image.csv", images)
    (output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reports": len(sources), "aggregate_rows": len(aggregates), "image_rows": len(images)}))


if __name__ == "__main__":
    main()
