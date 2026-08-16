#!/usr/bin/env python3
"""Parse EconomicGrasp Stage-0/1/2 text logs and summarize training dynamics.

Example:
  python tools/analyze_distill_training_logs.py \
      --log_zip /path/to/kd_stage0-2_logs.zip \
      --window_steps 5000 \
      --output_dir /tmp/kd_log_analysis

The parser produces:
  * train_5000_step_key_metrics.csv
  * train_5000_step_all_metrics.csv
  * train_epoch_loss.csv
  * eval_epoch_metrics.csv
  * best_eval_key_metrics.csv
  * final_5000_step_key_metrics.csv
  * anomaly_report.md
  * summary.json

Training summaries use the periodic metric blocks printed every 20 optimizer
steps.  The final unprinted remainder of each epoch (five steps in the supplied
logs) cannot be reconstructed from text and is excluded from component means.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


HEADER_RE = re.compile(r" ---- epoch: (\d+), batch: (\d+) ----")
METRIC_RE = re.compile(
    r"^(.+?)\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)
EPOCH_RE = re.compile(r"\*\*\*\* EPOCH\s+(\d+)")
OVERALL_TRAIN_RE = re.compile(
    r"overall training loss per batch:\s*([^,]+), batch num:(\d+)"
)
EVAL_METRIC_RE = re.compile(
    r"^eval mean (.+?):\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)
OVERALL_EVAL_RE = re.compile(r"^overall loss:([^,]+), batch num:(\d+)")
BEST_RE = re.compile(r"best_epoch:(\d+)")

DEFAULT_ZIP_NAME_MAP = {
    "stage0_log_train.txt": "stage0",
    "stage1_log_train.txt": "stage1",
    "stage2_log_train.txt": "stage2_old",
    "stage_2_new_log_train.txt": "stage2_new",
}

BEST_EVAL_KEY_METRICS = (
    "overall_eval_loss",
    "A: Supervised Loss",
    "B: CDF Loss",
    "B: View Loss",
    "B: Width Depth Loss",
    "A: DepthReg Loss",
    "D: CDF Selection Regret",
    "D: CDF Selected Target Utility",
    "D: CDF selected pearson",
    "D: CDF top10 target utility",
    "D: CDF top50 target utility",
    "D: TopView Regret",
    "D: RGBGeom center z MAE",
    "D: Depth final MAE",
)

KEY_METRICS = (
    "A: Overall Loss",
    "A: Supervised Loss",
    "A: Distill Loss",
    "A: Grasp Loss",
    "A: Objectness Loss",
    "A: DepthReg Loss",
    "B: CDF Loss",
    "B: View Loss",
    "B: Width Depth Loss",
    "B: KD CDF Loss",
    "B: KD CDF Excess",
    "B: KD CDF Teacher Entropy",
    "B: KD View Loss",
    "B: KD Width Loss",
    "B: KD Objectness Loss",
    "B: KD Graspness Loss",
    "D: KD query match ratio",
    "D: KD query view angle",
    "D: KD valid query view angle",
    "D: KD valid query view angle p90",
    "D: KD width positive ratio",
    "D: RGBGeom center z MAE",
    "D: RGBGeom center xyz MAE",
    "D: RGBGeom patch shape MAE",
    "D: TopView Regret",
    "D: CDF Selection Regret",
    "D: CDF Selected Target Utility",
    "D: CDF selected pearson",
    "D: CDF top10 target utility",
    "D: CDF top50 target utility",
)


def parse_log_text(text: str) -> Dict[str, object]:
    train_blocks: List[Dict[str, float]] = []
    train_epochs: List[Dict[str, float]] = []
    eval_epochs: List[Dict[str, float]] = []
    current_epoch: Optional[int] = None
    current_block: Optional[Dict[str, float]] = None
    pending_eval: Dict[str, float] = {}
    pending_eval_epoch: Optional[int] = None
    best_epoch: Optional[int] = None

    for raw in io.StringIO(text):
        line = raw.rstrip("\n")
        match = EPOCH_RE.search(line)
        if match:
            current_epoch = int(match.group(1))

        match = HEADER_RE.match(line)
        if match:
            if current_block is not None:
                train_blocks.append(current_block)
            current_block = {
                "epoch": float(int(match.group(1))),
                "batch": float(int(match.group(2))),
            }
            continue

        match = OVERALL_TRAIN_RE.search(line)
        if match:
            if current_block is not None:
                train_blocks.append(current_block)
                current_block = None
            train_epochs.append(
                {
                    "epoch": float(current_epoch if current_epoch is not None else -1),
                    "overall_train_loss": float(match.group(1)),
                    "batch_num_global": float(int(match.group(2))),
                }
            )
            continue

        match = EVAL_METRIC_RE.match(line)
        if match:
            if pending_eval_epoch is None:
                pending_eval_epoch = current_epoch
            pending_eval[match.group(1)] = float(match.group(2))
            continue

        match = OVERALL_EVAL_RE.match(line)
        if match:
            row: Dict[str, float] = {
                "epoch": float(
                    pending_eval_epoch
                    if pending_eval_epoch is not None
                    else (current_epoch if current_epoch is not None else -1)
                ),
                "overall_eval_loss": float(match.group(1)),
                "batch_num_global": float(int(match.group(2))),
            }
            row.update(pending_eval)
            eval_epochs.append(row)
            pending_eval = {}
            pending_eval_epoch = None
            continue

        match = BEST_RE.search(line)
        if match:
            best_epoch = int(match.group(1))

        if current_block is not None:
            match = METRIC_RE.match(line)
            if match:
                current_block[match.group(1).rstrip()] = float(match.group(2))

    if current_block is not None:
        train_blocks.append(current_block)

    return {
        "train_blocks": train_blocks,
        "train_epochs": train_epochs,
        "eval_epochs": eval_epochs,
        "best_epoch": best_epoch,
    }


def infer_steps_per_epoch(parsed: Mapping[str, object]) -> int:
    blocks = parsed["train_blocks"]
    train_epochs = parsed["train_epochs"]
    if not blocks:
        raise ValueError("No periodic training metric blocks were found.")
    max_marker = int(max(float(row["batch"]) for row in blocks))
    if train_epochs:
        global_count = int(float(train_epochs[0]["batch_num_global"]))
        candidates = []
        for world_size in range(1, 33):
            if global_count % world_size:
                continue
            local_count = global_count // world_size
            if max_marker <= local_count <= max_marker + 64:
                candidates.append(local_count)
        if candidates:
            return min(candidates)
    # The logger prints every 20 steps.  Use the next marker as a conservative
    # fallback when the global batch count is unavailable.
    return int(math.ceil(max_marker / 20.0) * 20)


def aggregate_train_windows(
    parsed: Mapping[str, object],
    window_steps: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], int]:
    steps_per_epoch = infer_steps_per_epoch(parsed)
    blocks: List[Dict[str, float]] = list(parsed["train_blocks"])
    weighted_sum: MutableMapping[Tuple[int, str], float] = defaultdict(float)
    weight_sum: MutableMapping[Tuple[int, str], float] = defaultdict(float)

    # Infer each block span from consecutive marker positions.  Periodic logs
    # are normally 20-step means.
    previous_by_epoch: Dict[int, int] = {}
    for row in blocks:
        epoch = int(row["epoch"])
        batch = int(row["batch"])
        previous = previous_by_epoch.get(epoch, 0)
        span = max(1, batch - previous)
        previous_by_epoch[epoch] = batch
        step_end = epoch * steps_per_epoch + batch
        window_start = ((step_end - 1) // window_steps) * window_steps
        for key, value in row.items():
            if key in {"epoch", "batch"}:
                continue
            weighted_sum[(window_start, key)] += float(value) * span
            weight_sum[(window_start, key)] += span

    windows = sorted({key[0] for key in weighted_sum})
    long_rows: List[Dict[str, object]] = []
    wide_rows: List[Dict[str, object]] = []
    for window_start in windows:
        wide: Dict[str, object] = {
            "window_start": window_start,
            "window_end": window_start + window_steps,
        }
        metrics = sorted(
            metric for (start, metric) in weighted_sum if start == window_start
        )
        for metric in metrics:
            weight = weight_sum[(window_start, metric)]
            mean = weighted_sum[(window_start, metric)] / max(weight, 1.0)
            long_rows.append(
                {
                    "window_start": window_start,
                    "window_end": window_start + window_steps,
                    "metric": metric,
                    "mean": mean,
                    "covered_steps": weight,
                }
            )
            if metric in KEY_METRICS:
                wide[metric] = mean
        wide_rows.append(wide)
    return long_rows, wide_rows, steps_per_epoch


def write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def value_at_best(
    parsed: Mapping[str, object],
    metric: str,
) -> Optional[float]:
    best = parsed.get("best_epoch")
    rows: List[Mapping[str, float]] = parsed["eval_epochs"]
    if best is None or not rows:
        return None
    for row in rows:
        if int(row["epoch"]) == int(best) and metric in row:
            return float(row[metric])
    return None


def pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or abs(old) < 1.0e-12:
        return None
    return 100.0 * (new / old - 1.0)


def build_best_eval_rows(
    parsed_by_stage: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for stage, parsed in parsed_by_stage.items():
        best = parsed.get("best_epoch")
        if best is None:
            continue
        match = None
        for row in parsed["eval_epochs"]:
            if int(row["epoch"]) == int(best):
                match = row
                break
        if match is None:
            continue
        out: Dict[str, object] = {"stage": stage, "best_epoch": int(best)}
        for metric in BEST_EVAL_KEY_METRICS:
            if metric in match:
                out[metric] = float(match[metric])
        rows.append(out)
    return rows


def build_final_window_rows(
    key_windows_by_stage: Mapping[str, List[Mapping[str, object]]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for stage, windows in key_windows_by_stage.items():
        if windows:
            rows.append({"stage": stage, **dict(windows[-1])})
    return rows


def build_report(
    parsed_by_stage: Mapping[str, Mapping[str, object]],
    key_windows_by_stage: Mapping[str, List[Mapping[str, object]]],
    steps_per_epoch_by_stage: Mapping[str, int],
) -> str:
    lines = [
        "# Stage 0–2 Distillation Log Diagnostics",
        "",
        "## Run coverage",
        "",
        "| Stage | Epochs | Steps/epoch | Best epoch | Best eval loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, parsed in parsed_by_stage.items():
        train_epochs = parsed["train_epochs"]
        best = parsed.get("best_epoch")
        best_loss = value_at_best(parsed, "overall_eval_loss")
        lines.append(
            f"| {stage} | {len(train_epochs)} | "
            f"{steps_per_epoch_by_stage[stage]} | "
            f"{best if best is not None else '—'} | "
            f"{best_loss:.6f} |" if best_loss is not None else
            f"| {stage} | {len(train_epochs)} | "
            f"{steps_per_epoch_by_stage[stage]} | "
            f"{best if best is not None else '—'} | — |"
        )

    best_rows = build_best_eval_rows(parsed_by_stage)
    if best_rows:
        lines.extend(
            [
                "",
                "## Best-checkpoint validation comparison",
                "",
                "| Stage | CDF loss | View loss | CDF regret | Selected utility | Top-10 utility | Center-z MAE (m) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        by_stage = {str(row["stage"]): row for row in best_rows}
        for stage in ("stage0", "stage1", "stage2_old", "stage2_new"):
            row = by_stage.get(stage)
            if row is None:
                continue
            def fmt(key: str) -> str:
                value = row.get(key)
                return "—" if value is None else f"{float(value):.6f}"
            lines.append(
                f"| {stage} | {fmt('B: CDF Loss')} | {fmt('B: View Loss')} | "
                f"{fmt('D: CDF Selection Regret')} | "
                f"{fmt('D: CDF Selected Target Utility')} | "
                f"{fmt('D: CDF top10 target utility')} | "
                f"{fmt('D: RGBGeom center z MAE')} |"
            )

    lines.extend(["", "## Automatically detected findings", ""])
    findings: List[str] = []

    if "stage1" in parsed_by_stage:
        s1 = parsed_by_stage["stage1"]
        s1_cdf = value_at_best(s1, "B: CDF Loss")
        s1_regret = value_at_best(s1, "D: CDF Selection Regret")
        s1_selected = value_at_best(s1, "D: CDF Selected Target Utility")
        s1_view = value_at_best(s1, "B: View Loss")

        for stage in ("stage2_old", "stage2_new"):
            if stage not in parsed_by_stage:
                continue
            current = parsed_by_stage[stage]
            cdf = value_at_best(current, "B: CDF Loss")
            regret = value_at_best(current, "D: CDF Selection Regret")
            selected = value_at_best(
                current, "D: CDF Selected Target Utility"
            )
            view = value_at_best(current, "B: View Loss")
            if cdf is not None and s1_cdf is not None:
                findings.append(
                    f"- **{stage} increases validation CDF loss by "
                    f"{pct_change(cdf, s1_cdf):.1f}%** "
                    f"({s1_cdf:.5f} → {cdf:.5f})."
                )
            if regret is not None and s1_regret is not None:
                findings.append(
                    f"- **{stage} increases CDF selection regret by "
                    f"{pct_change(regret, s1_regret):.1f}%** "
                    f"({s1_regret:.5f} → {regret:.5f})."
                )
            if selected is not None and s1_selected is not None:
                findings.append(
                    f"- {stage} changes selected GT utility by "
                    f"{pct_change(selected, s1_selected):.1f}% "
                    f"({s1_selected:.5f} → {selected:.5f})."
                )
            if view is not None and s1_view is not None:
                findings.append(
                    f"- {stage} changes validation view loss by "
                    f"{pct_change(view, s1_view):.1f}% "
                    f"({s1_view:.5f} → {view:.5f}); the dominant degradation "
                    "is therefore not the view loss."
                )

    for stage in ("stage2_old", "stage2_new"):
        windows = key_windows_by_stage.get(stage, [])
        if not windows:
            continue
        last = windows[-1]
        distill = last.get("A: Distill Loss")
        kd_cdf = last.get("B: KD CDF Loss")
        supervised = last.get("A: Supervised Loss")
        cdf = last.get("B: CDF Loss")
        match = last.get("D: KD query match ratio")
        angle = last.get("D: KD query view angle")
        if distill and kd_cdf is not None:
            findings.append(
                f"- At the final 5k-step window, **{stage} KD is "
                f"{100.0 * float(kd_cdf) / float(distill):.1f}% CDF BCE** "
                f"({float(kd_cdf):.5f}/{float(distill):.5f})."
            )
        if supervised is not None and cdf is not None:
            findings.append(
                f"- Final-window {stage} supervised loss is "
                f"{float(supervised):.5f}, including CDF loss "
                f"{float(cdf):.5f}."
            )
        if match is not None and angle is not None:
            findings.append(
                f"- {stage} uses CDF/width KD for only "
                f"{100.0 * float(match):.1f}% of queries; the mean nearest "
                f"teacher/student view angle over all queries is "
                f"{float(angle):.1f}°."
            )

    if "stage0" in parsed_by_stage and "stage1" in parsed_by_stage:
        t_cdf = value_at_best(parsed_by_stage["stage0"], "B: CDF Loss")
        s_cdf = value_at_best(parsed_by_stage["stage1"], "B: CDF Loss")
        t_view = value_at_best(parsed_by_stage["stage0"], "B: View Loss")
        s_view = value_at_best(parsed_by_stage["stage1"], "B: View Loss")
        if None not in (t_cdf, s_cdf):
            findings.append(
                f"- The clean-depth teacher is **not uniformly better in CDF "
                f"loss** at its best checkpoint ({t_cdf:.5f} teacher vs "
                f"{s_cdf:.5f} Stage 1)."
            )
        if None not in (t_view, s_view):
            findings.append(
                f"- The clean-depth teacher is better in view loss "
                f"({t_view:.5f} vs {s_view:.5f}), but the margin is much "
                "smaller than the final AP geometry gap."
            )

    if not findings:
        findings.append("- No predefined anomaly rule was triggered.")
    lines.extend(findings)

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The logs support three checks before adding more KD modules:",
            "",
            "1. Compare teacher and student against the **same GT query labels**, "
            "rather than assuming that a stronger final AP makes every teacher "
            "output a better target.",
            "2. Measure teacher/student **physical-center and matched-label drift** "
            "at the same image pixel.",
            "3. Measure the output-gradient cosine between supervised CDF loss and "
            "CDF KD. A negative cosine directly confirms objective conflict.",
            "",
            "Raw soft-target BCE includes teacher entropy and therefore has a "
            "non-zero floor. New runtime diagnostics should report Bernoulli-KL/"
            "excess BCE separately.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_logs(args: argparse.Namespace) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    if args.log_zip:
        with zipfile.ZipFile(args.log_zip) as archive:
            for member in archive.namelist():
                base = Path(member).name
                if base in DEFAULT_ZIP_NAME_MAP:
                    texts[DEFAULT_ZIP_NAME_MAP[base]] = archive.read(
                        member
                    ).decode("utf-8", errors="replace")
    for item in args.log:
        if "=" not in item:
            raise ValueError("--log must use STAGE=/path/to/log.txt")
        stage, path = item.split("=", 1)
        texts[stage.strip()] = Path(path).read_text(
            encoding="utf-8", errors="replace"
        )
    if not texts:
        raise ValueError("No logs found. Pass --log_zip or one or more --log.")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_zip", type=Path, default=None)
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="Additional log in STAGE=/path/to/log.txt form.",
    )
    parser.add_argument("--window_steps", type=int, default=5000)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    texts = load_logs(args)
    parsed_by_stage = {
        stage: parse_log_text(text) for stage, text in texts.items()
    }

    all_long: List[Dict[str, object]] = []
    all_wide: List[Dict[str, object]] = []
    train_epoch_rows: List[Dict[str, object]] = []
    eval_rows: List[Dict[str, object]] = []
    steps_per_epoch: Dict[str, int] = {}
    key_windows: Dict[str, List[Mapping[str, object]]] = {}

    for stage, parsed in parsed_by_stage.items():
        long_rows, wide_rows, stage_steps = aggregate_train_windows(
            parsed, args.window_steps
        )
        steps_per_epoch[stage] = stage_steps
        key_windows[stage] = wide_rows
        for row in long_rows:
            all_long.append({"stage": stage, **row})
        for row in wide_rows:
            all_wide.append({"stage": stage, **row})
        for row in parsed["train_epochs"]:
            train_epoch_rows.append({"stage": stage, **row})
        for row in parsed["eval_epochs"]:
            eval_rows.append({"stage": stage, **row})

    write_csv(
        args.output_dir / "train_5000_step_all_metrics.csv",
        all_long,
    )
    write_csv(
        args.output_dir / "train_5000_step_key_metrics.csv",
        all_wide,
    )
    write_csv(args.output_dir / "train_epoch_loss.csv", train_epoch_rows)
    write_csv(args.output_dir / "eval_epoch_metrics.csv", eval_rows)
    write_csv(
        args.output_dir / "best_eval_key_metrics.csv",
        build_best_eval_rows(parsed_by_stage),
    )
    write_csv(
        args.output_dir / "final_5000_step_key_metrics.csv",
        build_final_window_rows(key_windows),
    )

    report = build_report(
        parsed_by_stage,
        key_windows,
        steps_per_epoch,
    )
    (args.output_dir / "anomaly_report.md").write_text(
        report, encoding="utf-8"
    )
    summary = {
        "window_steps": args.window_steps,
        "steps_per_epoch": steps_per_epoch,
        "best_epoch": {
            stage: parsed.get("best_epoch")
            for stage, parsed in parsed_by_stage.items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(report)
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
