#!/usr/bin/env python3
"""Compare matched MGF no-ranking vs ranking training runs.

Expected input directories are the two TRAIN_ROOT folders, each containing:
  protocol.json
  metrics.json

The script fails if the protocols differ in anything other than ranking_weight.
It summarizes epoch-20/latest metrics, best discriminability over training, and
writes a per-epoch TSV. If official latest-checkpoint AP summaries exist under
the parent WORK_ROOT, they are included opportunistically.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


VAL_KEYS = [
    "cdf",
    "base_cdf",
    "field_cdf_any_success_auroc64",
    "field_cdf_any_success_auprc64",
    "field_cdf_any_success_auprc_lift64",
    "field_cdf_pos_neg_gap",
    "field_cdf_utility_pearson",
    "field_cdf_utility_pred_mean",
    "field_cdf_utility_target_mean",
    "ranking",
    "ranking_weighted",
    "ranking_informative_query_fraction",
    "ranking_selection_regret",
    "ranking_top1_best_hit",
    "ranking_selected_target_utility",
    "ranking_oracle_target_utility",
    "width_label_m_mean",
    "width_pred_decoded_m_mean",
    "depth_l1",
    "profile_mean_l1",
]

# metric -> optimization direction for "best over epochs"
BEST_RULES = {
    "field_cdf_any_success_auroc64": "max",
    "field_cdf_any_success_auprc64": "max",
    "field_cdf_any_success_auprc_lift64": "max",
    "field_cdf_pos_neg_gap": "max",
    "field_cdf_utility_pearson": "max",
    "ranking_selection_regret": "min",
    "ranking_top1_best_hit": "max",
    "cdf": "min",
}


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_number(value: Any, context: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise TypeError(f"{context} is not numeric: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{context} is non-finite: {out}")
    return out


def normalize_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    p = copy.deepcopy(protocol)
    loss = p.get("loss_weights")
    if not isinstance(loss, dict):
        raise KeyError("protocol.loss_weights is missing or invalid")
    loss.pop("ranking_weight", None)
    return p


def validate_protocols(no: Dict[str, Any], yes: Dict[str, Any]) -> None:
    no_rank = finite_number(
        no.get("loss_weights", {}).get("ranking_weight"),
        "no-ranking ranking_weight",
    )
    yes_rank = finite_number(
        yes.get("loss_weights", {}).get("ranking_weight"),
        "ranking ranking_weight",
    )
    if no_rank != 0.0:
        raise RuntimeError(
            f"No-ranking run must use ranking_weight=0, got {no_rank}"
        )
    if yes_rank <= 0.0:
        raise RuntimeError(
            f"Ranking run must use ranking_weight>0, got {yes_rank}"
        )

    if normalize_protocol(no) != normalize_protocol(yes):
        # Emit focused high-value differences rather than accepting a confound.
        keys = sorted(set(no) | set(yes))
        diffs = []
        for key in keys:
            a, b = no.get(key), yes.get(key)
            if key == "loss_weights" and isinstance(a, dict) and isinstance(b, dict):
                aa, bb = dict(a), dict(b)
                aa.pop("ranking_weight", None)
                bb.pop("ranking_weight", None)
                if aa != bb:
                    diffs.append((key, aa, bb))
            elif a != b:
                diffs.append((key, a, b))
        raise RuntimeError(
            "Ablation protocols differ outside ranking_weight:\n"
            + "\n".join(
                f"  {key}: no-ranking={a!r}, ranking={b!r}"
                for key, a, b in diffs
            )
        )

    if no.get("init_checkpoint", None) != "":
        raise RuntimeError(
            "This controlled ablation requires init_checkpoint='' for both runs"
        )
    if bool(no.get("partial_run", True)):
        raise RuntimeError("Formal ablation protocol is marked partial_run=true")
    if abs(float(no.get("sample_fraction", -1)) - 0.1) > 1e-12:
        raise RuntimeError(
            f"Expected sample_fraction=0.1, got {no.get('sample_fraction')}"
        )
    if int(no.get("train_frames", -1)) != 2600:
        raise RuntimeError(
            f"Expected 2600 train frames, got {no.get('train_frames')}"
        )
    if int(no.get("val_frames", -1)) != 780:
        raise RuntimeError(
            f"Expected 780 Seen-val frames, got {no.get('val_frames')}"
        )


def validate_history(
    history: List[Dict[str, Any]],
    expected_epochs: int,
    name: str,
) -> None:
    if not isinstance(history, list) or not history:
        raise RuntimeError(f"{name} metrics.json is empty")
    epochs = [int(row["epoch"]) for row in history]
    expected = list(range(expected_epochs))
    if epochs != expected:
        raise RuntimeError(
            f"{name} expected epochs {expected[0]}..{expected[-1]}, got {epochs}"
        )
    for row in history:
        if "train" not in row or "validation" not in row:
            raise KeyError(f"{name} epoch {row.get('epoch')} lacks train/validation")
        for split in ("train", "validation"):
            stats = row[split]
            for key in VAL_KEYS:
                if key not in stats:
                    raise KeyError(
                        f"{name} epoch {row['epoch']} {split} missing metric {key}"
                    )
                finite_number(
                    stats[key],
                    f"{name} epoch {row['epoch']} {split}.{key}",
                )


def best_epoch(history: List[Dict[str, Any]], key: str, direction: str):
    rows = []
    for row in history:
        value = finite_number(
            row["validation"][key],
            f"epoch {row['epoch']} validation.{key}",
        )
        rows.append((int(row["epoch"]), value))
    if direction == "max":
        return max(rows, key=lambda x: x[1])
    if direction == "min":
        return min(rows, key=lambda x: x[1])
    raise ValueError(direction)


def maybe_official_ap(train_dir: Path):
    work_root = train_dir.parent
    out = {}
    for split in ("test_seen", "test_similar", "test_novel"):
        path = (
            work_root
            / "test_latest"
            / "official"
            / split
            / "summary.json"
        )
        if path.is_file():
            summary = load_json(path)
            out[split] = summary.get("reported_ap")
    return out


def fmt(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{float(v):.6f}"
    return str(v)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-ranking", required=True, type=Path)
    p.add_argument("--ranking", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--expected-epochs", type=int, default=20)
    args = p.parse_args()

    if args.expected_epochs <= 0:
        raise ValueError("--expected-epochs must be >0")

    no_protocol = load_json(args.no_ranking / "protocol.json")
    yes_protocol = load_json(args.ranking / "protocol.json")
    validate_protocols(no_protocol, yes_protocol)

    no_hist = load_json(args.no_ranking / "metrics.json")
    yes_hist = load_json(args.ranking / "metrics.json")
    validate_history(no_hist, args.expected_epochs, "no-ranking")
    validate_history(yes_hist, args.expected_epochs, "ranking")

    final_no = no_hist[-1]["validation"]
    final_yes = yes_hist[-1]["validation"]

    best = {"no_ranking": {}, "ranking": {}}
    for key, direction in BEST_RULES.items():
        e, v = best_epoch(no_hist, key, direction)
        best["no_ranking"][key] = {
            "epoch": e,
            "value": v,
            "direction": direction,
        }
        e, v = best_epoch(yes_hist, key, direction)
        best["ranking"][key] = {
            "epoch": e,
            "value": v,
            "direction": direction,
        }

    final_table = {}
    for key in VAL_KEYS:
        a = finite_number(final_no[key], f"final no-ranking {key}")
        b = finite_number(final_yes[key], f"final ranking {key}")
        final_table[key] = {
            "no_ranking": a,
            "ranking": b,
            "delta_ranking_minus_no": b - a,
        }

    result = {
        "contract": {
            "sample_fraction": no_protocol["sample_fraction"],
            "train_frames": no_protocol["train_frames"],
            "seen_val_frames": no_protocol["val_frames"],
            "sampling_sha256": no_protocol["sampling_sha256"],
            "seed": no_protocol["seed"],
            "init_checkpoint": no_protocol["init_checkpoint"],
            "world_size": no_protocol["optimizer"]["world_size"],
            "batch_per_gpu": no_protocol["optimizer"]["batch_per_gpu"],
            "effective_batch": no_protocol["optimizer"]["effective_batch"],
            "epochs": args.expected_epochs,
            "ranking_temperature": no_protocol["loss_weights"][
                "ranking_temperature"
            ],
            "no_ranking_weight": no_protocol["loss_weights"]["ranking_weight"],
            "ranking_weight": yes_protocol["loss_weights"]["ranking_weight"],
        },
        "epoch20_validation": final_table,
        "best_validation": best,
        "official_latest_ap": {
            "no_ranking": maybe_official_ap(args.no_ranking),
            "ranking": maybe_official_ap(args.ranking),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    tsv_keys = [
        "cdf",
        "field_cdf_any_success_auroc64",
        "field_cdf_any_success_auprc64",
        "field_cdf_any_success_auprc_lift64",
        "field_cdf_pos_neg_gap",
        "field_cdf_utility_pearson",
        "ranking_informative_query_fraction",
        "ranking_selection_regret",
        "ranking_top1_best_hit",
        "depth_l1",
        "width_pred_decoded_m_mean",
    ]
    with (args.output_dir / "per_epoch.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f, delimiter="\t")
        header = ["epoch"]
        for key in tsv_keys:
            header.extend(
                [
                    f"no_{key}",
                    f"rank_{key}",
                    f"delta_{key}",
                ]
            )
        writer.writerow(header)
        for no_row, yes_row in zip(no_hist, yes_hist):
            row = [int(no_row["epoch"])]
            for key in tsv_keys:
                a = float(no_row["validation"][key])
                b = float(yes_row["validation"][key])
                row.extend([a, b, b - a])
            writer.writerow(row)

    md = []
    md.append("# MGF Ranking Ablation Comparison")
    md.append("")
    c = result["contract"]
    md.append(
        f"Matched protocol: 10% GraspNet ({c['train_frames']} train / "
        f"{c['seen_val_frames']} Seen-val), seed={c['seed']}, "
        f"{c['world_size']} GPUs x batch {c['batch_per_gpu']} = "
        f"effective batch {c['effective_batch']}, "
        f"init_checkpoint={c['init_checkpoint']!r}, "
        f"ranking temperature={c['ranking_temperature']}."
    )
    md.append("")
    md.append("## Epoch 20 (epoch index 19) Seen validation")
    md.append("")
    md.append("| Metric | No ranking | Ranking | Delta (rank - no) |")
    md.append("|---|---:|---:|---:|")
    display_keys = [
        "cdf",
        "field_cdf_any_success_auroc64",
        "field_cdf_any_success_auprc64",
        "field_cdf_any_success_auprc_lift64",
        "field_cdf_pos_neg_gap",
        "field_cdf_utility_pearson",
        "ranking_informative_query_fraction",
        "ranking_selection_regret",
        "ranking_top1_best_hit",
        "depth_l1",
        "width_label_m_mean",
        "width_pred_decoded_m_mean",
    ]
    for key in display_keys:
        x = final_table[key]
        md.append(
            f"| {key} | {fmt(x['no_ranking'])} | {fmt(x['ranking'])} | "
            f"{fmt(x['delta_ranking_minus_no'])} |"
        )

    md.append("")
    md.append("## Best Seen-validation value across 20 epochs")
    md.append("")
    md.append(
        "| Metric | Direction | No ranking (epoch,value) | "
        "Ranking (epoch,value) |"
    )
    md.append("|---|---|---:|---:|")
    for key, direction in BEST_RULES.items():
        a = best["no_ranking"][key]
        b = best["ranking"][key]
        md.append(
            f"| {key} | {direction} | ({a['epoch']},{fmt(a['value'])}) | "
            f"({b['epoch']},{fmt(b['value'])}) |"
        )

    aps = result["official_latest_ap"]
    if aps["no_ranking"] or aps["ranking"]:
        md.append("")
        md.append("## Official AP from latest checkpoint (when available)")
        md.append("")
        md.append("| Split | No ranking | Ranking |")
        md.append("|---|---|---|")
        for split in ("test_seen", "test_similar", "test_novel"):
            md.append(
                f"| {split} | {aps['no_ranking'].get(split, 'N/A')} | "
                f"{aps['ranking'].get(split, 'N/A')} |"
            )

    md.append("")
    md.append(
        "Use epoch-20/latest as the primary controlled comparison. "
        "The trainer's checkpoint_best is selected by CDF BCE and can favor "
        "a constant-prior solution, so it is not the primary ranking-ablation result."
    )
    md.append("")
    (args.output_dir / "comparison.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )

    print("\n".join(md))


if __name__ == "__main__":
    main()
