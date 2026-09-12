#!/usr/bin/env python3
"""Analyze P5 safe-vs-repairable evidence separability.

Inputs are four extraction directories produced by diagnose_p5_separability.py:
train/native, train/corrupt, test_seen/native, test_seen/corrupt.  The analysis
fits frozen-feature linear probes, reports AUROC/AUPRC and low-intervention
precision, and simulates oracle/learned activation gates using the already
predicted P5 repair vector.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn

from utils.p5_separability import (
    FEATURE_NAMES, classification_summary, gated_query_metrics,
    threshold_for_safe_fpr,
)


ARRAY_KEYS = (
    *FEATURE_NAMES, "safe", "repairable", "need_repair", "beneficial",
    "gain_m", "native_violation_m", "repaired_violation_m", "native_safe",
    "repaired_safe", "pred_delta_abs_m", "query_idx", "frame_idx", "scene_id",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_native", required=True)
    p.add_argument("--train_corrupt", required=True)
    p.add_argument("--eval_native", required=True)
    p.add_argument("--eval_corrupt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--probe_device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--probe_epochs", type=int, default=15)
    p.add_argument("--probe_lr", type=float, default=0.02)
    p.add_argument("--probe_batch_size", type=int, default=8192)
    p.add_argument("--probe_weight_decay", type=float, default=1e-4)
    p.add_argument("--max_train_queries_per_condition", type=int, default=250000)
    p.add_argument("--seed", type=int, default=1931)
    return p.parse_args()


def load_shards(root: str):
    path = Path(root)
    files = sorted(path.glob("rank*_chunk*.npz"))
    if not files:
        raise FileNotFoundError(f"No diagnostic shards found under {path}")
    parts = {k: [] for k in ARRAY_KEYS}
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            missing = [k for k in ARRAY_KEYS if k not in z]
            if missing:
                raise KeyError(f"{f} missing arrays {missing}")
            for k in ARRAY_KEYS:
                parts[k].append(np.asarray(z[k]))
    arrays = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    n = len(arrays["safe"])
    if any(len(arrays[k]) != n for k in ARRAY_KEYS):
        raise RuntimeError(f"Inconsistent shard lengths in {path}")
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    return arrays, summary


def subsample(arrays, max_n: int, seed: int):
    n = len(arrays["safe"])
    if max_n <= 0 or n <= max_n:
        return arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    idx.sort()
    return {k: v[idx] for k, v in arrays.items()}


def combine(a, b):
    return {k: np.concatenate((a[k], b[k]), axis=0) for k in ARRAY_KEYS}


class LinearProbe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(int(dim), 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


def fit_probe(X: np.ndarray, y: np.ndarray, args, device: torch.device, seed: int):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(y) < 100 or y.min() == y.max():
        raise ValueError("Probe training requires at least 100 samples and both classes.")
    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-5)
    X = (X - mean) / std

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = LinearProbe(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.probe_lr),
                            weight_decay=float(args.probe_weight_decay))
    npos = max(float(y.sum()), 1.0)
    nneg = max(float(len(y) - y.sum()), 1.0)
    pos_weight = torch.tensor(nneg / npos, dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng = np.random.default_rng(seed)
    bs = int(args.probe_batch_size)
    model.train()
    for _epoch in range(int(args.probe_epochs)):
        order = rng.permutation(len(y))
        for start in range(0, len(y), bs):
            ids = order[start:start + bs]
            xb = torch.from_numpy(X[ids]).to(device, non_blocking=False)
            yb = torch.from_numpy(y[ids]).to(device, non_blocking=False)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    weight = model.fc.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
    bias = float(model.fc.bias.detach().cpu().item())
    return {"mean": mean, "std": std, "weight": weight, "bias": bias}


def probe_score(probe, X):
    X = np.asarray(X, dtype=np.float32)
    z = ((X - probe["mean"]) / probe["std"]) @ probe["weight"] + probe["bias"]
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def gate_rows(dataset_name, arrays, task, feature, regime, scores, train_y, train_scores):
    rows = []
    baseline = gated_query_metrics(arrays, np.zeros(len(scores), dtype=bool))
    p5 = gated_query_metrics(arrays, np.ones(len(scores), dtype=bool))
    oracle_need = gated_query_metrics(arrays, arrays["repairable"].astype(bool))
    oracle_benefit = gated_query_metrics(arrays, arrays["beneficial"].astype(bool))
    for name, m in (("native", baseline), ("p5_all", p5),
                    ("oracle_need", oracle_need), ("oracle_benefit", oracle_benefit)):
        rows.append({"dataset": dataset_name, "task": task, "feature": feature,
                     "train_regime": regime, "gate": name, **m})

    order = np.argsort(-scores, kind="mergesort")
    for frac in (0.05, 0.10, 0.20):
        k = max(1, int(np.ceil(frac * len(scores))))
        active = np.zeros(len(scores), dtype=bool)
        active[order[:k]] = True
        rows.append({"dataset": dataset_name, "task": task, "feature": feature,
                     "train_regime": regime, "gate": f"top_{int(frac*100)}pct",
                     **gated_query_metrics(arrays, active)})
    for fpr in (0.01, 0.02, 0.05):
        threshold = threshold_for_safe_fpr(train_y, train_scores, fpr)
        active = scores >= threshold
        rows.append({"dataset": dataset_name, "task": task, "feature": feature,
                     "train_regime": regime, "gate": f"train_neg_fpr_{int(fpr*100)}pct",
                     "threshold": threshold, **gated_query_metrics(arrays, active)})
    return rows


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.probe_epochs <= 0 or args.probe_batch_size <= 0 or args.probe_lr <= 0:
        raise ValueError("Invalid probe optimization settings.")
    if args.probe_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--probe_device cuda requested but CUDA is unavailable.")
    device = torch.device(
        "cuda" if args.probe_device == "cuda" or
        (args.probe_device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tn, tn_meta = load_shards(args.train_native)
    tc, tc_meta = load_shards(args.train_corrupt)
    en, en_meta = load_shards(args.eval_native)
    ec, ec_meta = load_shards(args.eval_corrupt)
    tn = subsample(tn, args.max_train_queries_per_condition, args.seed + 1)
    tc = subsample(tc, args.max_train_queries_per_condition, args.seed + 2)
    mixed = combine(tn, tc)

    train_sets = {"native": tn, "corrupt": tc, "mixed": mixed}
    eval_sets = {"seen_native": en, "seen_corrupt": ec}
    probe_rows = []
    gate_results = []
    saved_probes = {}
    task_ids = {"need_repair": 1, "beneficial": 2}
    feature_ids = {name: i + 1 for i, name in enumerate(FEATURE_NAMES)}
    regime_ids = {"native": 1, "corrupt": 2, "mixed": 3}

    for task in ("need_repair", "beneficial"):
        for feature in FEATURE_NAMES:
            for regime, train in train_sets.items():
                y_train = train[task].astype(np.uint8)
                probe_seed = (
                    int(args.seed)
                    + 10000 * task_ids[task]
                    + 100 * feature_ids[feature]
                    + regime_ids[regime]
                )
                probe = fit_probe(train[feature], y_train, args, device, seed=probe_seed)
                train_score = probe_score(probe, train[feature])
                probe_key = f"{task}/{feature}/{regime}"
                saved_probes[probe_key] = {
                    "mean": probe["mean"].tolist(), "std": probe["std"].tolist(),
                    "weight": probe["weight"].tolist(), "bias": probe["bias"],
                    "seed": probe_seed,
                }
                for dataset_name, eval_arrays in eval_sets.items():
                    score = probe_score(probe, eval_arrays[feature])
                    metrics = classification_summary(eval_arrays[task], score)
                    probe_rows.append({
                        "dataset": dataset_name, "task": task, "feature": feature,
                        "train_regime": regime, **metrics,
                    })
                    gate_results.extend(gate_rows(
                        dataset_name, eval_arrays, task, feature, regime, score,
                        y_train, train_score,
                    ))

    with (out / "probe_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in probe_rows for k in r}))
        writer.writeheader(); writer.writerows(probe_rows)
    with (out / "gate_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in gate_results for k in r}))
        writer.writeheader(); writer.writerows(gate_results)
    (out / "linear_probes.json").write_text(json.dumps(saved_probes), encoding="utf-8")

    report = {
        "experiment": "p5-safe-repairable-separability-v1",
        "probe_device": str(device),
        "probe_epochs": int(args.probe_epochs),
        "max_train_queries_per_condition": int(args.max_train_queries_per_condition),
        "counts": {
            "train_native": len(tn["safe"]), "train_corrupt": len(tc["safe"]),
            "eval_native": len(en["safe"]), "eval_corrupt": len(ec["safe"]),
        },
        "source_summaries": {
            "train_native": tn_meta, "train_corrupt": tc_meta,
            "eval_native": en_meta, "eval_corrupt": ec_meta,
        },
        "probe_metrics_csv": str((out / "probe_metrics.csv").resolve()),
        "gate_metrics_csv": str((out / "gate_metrics.csv").resolve()),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# P5 safe-vs-repairable separability diagnostic", "",
        "## Linear probe: mixed train -> Seen", "",
        "| Task | Feature | Native AUROC | Native AUPRC | Corrupt AUROC | Corrupt AUPRC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lookup = {(r["task"], r["feature"], r["train_regime"], r["dataset"]): r for r in probe_rows}
    for task in ("need_repair", "beneficial"):
        for feature in FEATURE_NAMES:
            a = lookup[(task, feature, "mixed", "seen_native")]
            b = lookup[(task, feature, "mixed", "seen_corrupt")]
            lines.append(
                f"| {task} | {feature} | {a['auroc']:.4f} | {a['auprc']:.4f} | "
                f"{b['auroc']:.4f} | {b['auprc']:.4f} |"
            )
    lines += ["", "## Oracle activation bounds", "",
              "These use the existing P5 repair vector; only activation changes.", ""]
    for dataset_name in ("seen_native", "seen_corrupt"):
        subset = [r for r in gate_results if r["dataset"] == dataset_name and
                  r["task"] == "need_repair" and r["feature"] == "F0" and
                  r["train_regime"] == "mixed" and r["gate"] in
                  ("native", "p5_all", "oracle_need", "oracle_benefit")]
        lines.append(f"### {dataset_name}")
        lines.append("")
        lines.append("| Gate | Active | Set violation (mm) | Within-safe | Safe no-harm | Repairable capture |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in subset:
            lines.append(
                f"| {r['gate']} | {100*r['active_ratio']:.2f}% | {1000*r['set_violation_m']:.3f} | "
                f"{100*r['within_safe']:.2f}% | {100*r['safe_noharm']:.2f}% | "
                f"{100*r['repairable_capture']:.2f}% |"
            )
        lines.append("")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[P5-DIAG] wrote {out / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
