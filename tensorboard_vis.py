#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot RayMix-related TensorBoard scalars only.

Supports TensorBoard tags like:
  train_A: ...
  train_B: ...
  train_C: ...
  train_D: ...

Example:
    python plot_tb_raymix.py \
        --input /path/to/tb_logdir \
        --output /path/to/raymix_summary.png

Optional:
    python plot_tb_raymix.py \
        --input /path/to/tb_logdir \
        --output raymix_summary.png \
        --include "^(train_[CD]: )?(RayMix|Group Alpha|Evidence|Valid Points)" \
        --smooth 0.9 \
        --show_raw \
        --print_tags
"""

import os
import re
import math
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_files(path: str):
    if os.path.isfile(path):
        return [path]
    event_files = []
    for root, _, files in os.walk(path):
        for f in files:
            if "tfevents" in f:
                event_files.append(os.path.join(root, f))
    event_files.sort()
    return event_files


def load_scalars_from_event_file(event_file: str):
    ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
    ea.Reload()

    out = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        out[tag] = {
            "step": np.array([e.step for e in events], dtype=np.int64),
            "value": np.array([e.value for e in events], dtype=np.float64),
        }
    return out


def merge_scalar_dicts(list_of_scalar_dicts):
    merged = defaultdict(lambda: {"step": [], "value": []})

    for d in list_of_scalar_dicts:
        for tag, x in d.items():
            merged[tag]["step"].append(x["step"])
            merged[tag]["value"].append(x["value"])

    final = {}
    for tag, x in merged.items():
        step = np.concatenate(x["step"], axis=0)
        value = np.concatenate(x["value"], axis=0)

        order = np.argsort(step, kind="stable")
        step = step[order]
        value = value[order]

        # dedup: keep last value for repeated step
        uniq_step = []
        uniq_value = []
        last_s = None
        for s, v in zip(step, value):
            if last_s is not None and s == last_s:
                uniq_value[-1] = v
            else:
                uniq_step.append(s)
                uniq_value.append(v)
                last_s = s

        final[tag] = {
            "step": np.array(uniq_step, dtype=np.int64),
            "value": np.array(uniq_value, dtype=np.float64),
        }
    return final


def ema_smooth(y: np.ndarray, alpha: float):
    if len(y) == 0:
        return y
    out = np.empty_like(y, dtype=np.float64)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * y[i]
    return out


def clean_title(tag: str):
    # 去掉常见 train_ 前缀，让图标题更短
    tag = re.sub(r"^train_", "", tag)
    tag = re.sub(r"^val_", "val ", tag)
    return tag


def default_raymix_sort_key(tag: str):
    priority = [
        "train_C: Valid Points",
        "train_D: RayMix BestDist",
        "train_D: RayMix Valid Ratio",
        "train_D: Group Alpha Top1",
        "train_D: Group Alpha Entropy",
        "train_D: Group Alpha Changed Ratio",
        "train_D: Group Evidence Margin",
    ]
    for i, p in enumerate(priority):
        if tag == p:
            return (0, i, tag)

    # fallback priority by keyword
    fallback = {
        "Valid Points": 10,
        "RayMix BestDist": 11,
        "RayMix Valid Ratio": 12,
        "Group Alpha Top1": 13,
        "Group Alpha Entropy": 14,
        "Group Alpha Changed Ratio": 15,
        "Group Evidence Margin": 16,
        "RayMix": 30,
        "Group Alpha": 31,
        "Evidence": 32,
    }
    for k, rank in fallback.items():
        if k in tag:
            return (1, rank, tag)

    return (99, 999, tag)


def filter_tags(tags, include_pattern=None, exclude_pattern=None):
    include_re = re.compile(include_pattern) if include_pattern else None
    exclude_re = re.compile(exclude_pattern) if exclude_pattern else None

    out = []
    for tag in tags:
        if include_re is not None and include_re.search(tag) is None:
            continue
        if exclude_re is not None and exclude_re.search(tag) is not None:
            continue
        out.append(tag)
    return out


def plot_scalar_grid(
    scalar_data,
    tags,
    output_path,
    smooth_alpha=0.9,
    max_cols=3,
    show_raw=True,
):
    n = len(tags)
    if n == 0:
        raise ValueError("No tags selected for plotting.")

    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.2 * ncols, 3.5 * nrows),
        squeeze=False
    )

    for ax in axes.flatten():
        ax.set_visible(False)

    for i, tag in enumerate(tags):
        ax = axes[i // ncols][i % ncols]
        ax.set_visible(True)

        step = scalar_data[tag]["step"]
        value = scalar_data[tag]["value"]

        if len(step) == 0:
            ax.set_title(clean_title(tag), fontsize=10)
            ax.text(0.5, 0.5, "empty", ha="center", va="center")
            continue

        if show_raw:
            ax.plot(step, value, linewidth=0.9, alpha=0.30, label="raw")

        if smooth_alpha is not None and 0.0 <= smooth_alpha < 1.0 and len(value) >= 2:
            smoothed = ema_smooth(value, smooth_alpha)
            ax.plot(step, smoothed, linewidth=1.6, label=f"ema({smooth_alpha:.2f})")
        else:
            ax.plot(step, value, linewidth=1.6, label="value")

        ax.set_title(clean_title(tag), fontsize=10)
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("value", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=8)

        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle("RayMix-related Training Scalars", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="TensorBoard log directory or a single event file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output image path, e.g. raymix_summary.png")
    parser.add_argument("--include", type=str, default=None,
                        help="Regex to include tags. If omitted, use RayMix-related defaults for train_A|B|C|D style tags.")
    parser.add_argument("--exclude", type=str, default=None,
                        help="Regex to exclude tags.")
    parser.add_argument("--smooth", type=float, default=0.9,
                        help="EMA smoothing alpha in [0,1). Default: 0.9")
    parser.add_argument("--max_cols", type=int, default=3,
                        help="Max subplot columns. Default: 3")
    parser.add_argument("--show_raw", action="store_true",
                        help="Also draw raw curves.")
    parser.add_argument("--print_tags", action="store_true",
                        help="Print selected tags.")
    args = parser.parse_args()

    event_files = find_event_files(args.input)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {args.input}")

    print(f"[INFO] Found {len(event_files)} event file(s).")
    for f in event_files:
        print(f"  - {f}")

    all_scalar_dicts = []
    for f in event_files:
        try:
            all_scalar_dicts.append(load_scalars_from_event_file(f))
        except Exception as e:
            print(f"[WARN] Failed to read {f}: {e}")

    if not all_scalar_dicts:
        raise RuntimeError("Failed to load any scalar data.")

    scalar_data = merge_scalar_dicts(all_scalar_dicts)
    all_tags = sorted(scalar_data.keys(), key=default_raymix_sort_key)

    # 默认只保留 RayMix 相关的 train_A|B|C|D 变量
    if args.include is None:
        args.include = (
            r"^train_[ABCD]: ("
            r"Valid Points|"
            r"RayMix BestDist|"
            r"RayMix Valid Ratio|"
            r"Group Alpha Top1|"
            r"Group Alpha Entropy|"
            r"Group Alpha Changed Ratio|"
            r"Group Evidence Margin"
            r")$"
        )

    tags = filter_tags(all_tags, include_pattern=args.include, exclude_pattern=args.exclude)

    if args.print_tags:
        print("[INFO] Selected tags:")
        for t in tags:
            print(" ", t)

    if len(tags) == 0:
        print("[INFO] Available tags:")
        for t in all_tags:
            print(" ", t)
        raise RuntimeError("No scalar tags matched the include/exclude rules.")

    plot_scalar_grid(
        scalar_data=scalar_data,
        tags=tags,
        output_path=args.output,
        smooth_alpha=args.smooth,
        max_cols=args.max_cols,
        show_raw=args.show_raw,
    )

    print(f"[INFO] Saved figure to: {args.output}")


if __name__ == "__main__":
    main()