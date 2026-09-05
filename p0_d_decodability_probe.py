#!/usr/bin/env python3
"""P0-D: layer-wise privileged-information decodability probes.

The script freezes all EconomicGrasp features by operating on an existing
paired teacher/student diagnosis cache. For every available student layer it
fits:

* a linear probe;
* a two-layer MLP probe.

Targets are (1) GT CDF, (2) teacher CDF, and (3) whether the teacher is better
than the student on the corresponding query. When a row-aligned teacher feature
exists, it also fits a PCA-space feature-regression probe.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pkd_p0.common import CDF_THRESHOLDS, atomic_json_dump, seed_everything
from pkd_p0.paired_cache import (
    as_numpy,
    discover_payload_files,
    find_value,
    load_aliases,
    load_payload,
    move_threshold_last,
)


@dataclass
class LayerRows:
    layer: str
    feature: np.ndarray
    gt_target: np.ndarray
    teacher_target: np.ndarray
    teacher_better: np.ndarray
    scene_id: np.ndarray
    teacher_feature: Optional[np.ndarray]
    granularity: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--diagnosis_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mapping_json", default="")
    p.add_argument("--layers", default="seed,pre_view,selected_view,local,pre_cdf")
    p.add_argument("--val_scene_start", type=int, default=90)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--max_rows_per_layer", type=int, default=500000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--pca_dim", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out.astype(np.float32)


def friction_to_cdf(friction: np.ndarray) -> np.ndarray:
    f = np.asarray(friction, dtype=np.float32)
    thresholds = np.asarray(CDF_THRESHOLDS, dtype=np.float32)
    return ((f[..., None] > 0.0) & (f[..., None] <= thresholds)).astype(np.float32)


def scalar_scene_id(payload: Mapping[str, Any], aliases: Mapping[str, Sequence[str]], path: Path) -> int:
    _, value = find_value(payload, aliases["scene_id"])
    if value is not None:
        return int(as_numpy(value).reshape(-1)[0])
    match = re.search(r"scene[_-]?(\d{4})", str(path))
    if match:
        return int(match.group(1))
    raise KeyError(f"Cannot determine scene id for {path}")


def cdf_tensors(payload: Mapping[str, Any], aliases: Mapping[str, Sequence[str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, student_value = find_value(payload, aliases["student_logits"])
    _, teacher_value = find_value(payload, aliases["teacher_logits"])
    if student_value is None or teacher_value is None:
        raise KeyError("missing student/teacher CDF logits")
    student = move_threshold_last(as_numpy(student_value)).astype(np.float32)
    teacher = move_threshold_last(as_numpy(teacher_value)).astype(np.float32)
    if student.shape != teacher.shape:
        raise ValueError(f"student/teacher CDF shapes differ: {student.shape} vs {teacher.shape}")
    _, gt_value = find_value(payload, aliases["gt_cdf"])
    if gt_value is not None:
        gt = move_threshold_last(as_numpy(gt_value)).astype(np.float32)
    else:
        _, friction_value = find_value(payload, aliases["friction"])
        if friction_value is None:
            raise KeyError("missing GT CDF/friction")
        gt = friction_to_cdf(as_numpy(friction_value))
    try:
        gt = np.broadcast_to(gt, student.shape).astype(np.float32)
    except ValueError as exc:
        raise ValueError(f"GT shape {gt.shape} cannot broadcast to CDF {student.shape}") from exc
    return student, teacher, gt


def infer_cdf_axes(cdf: np.ndarray) -> Tuple[int, Optional[int]]:
    """Return threshold axis (last) and likely depth axis."""
    if cdf.shape[-1] != len(CDF_THRESHOLDS):
        raise ValueError(f"CDF threshold axis is not last: {cdf.shape}")
    depth_candidates = [axis for axis, size in enumerate(cdf.shape[:-1]) if size == 4]
    # Current tensors are normally [..., A=12, D=4, T=6], making the last D
    # candidate the least ambiguous.
    depth_axis = depth_candidates[-1] if depth_candidates else None
    return cdf.ndim - 1, depth_axis


def feature_matrix(array: np.ndarray) -> Tuple[np.ndarray, int]:
    """Flatten a feature tensor and return [tokens,C] plus token count.

    Prefer a final channel axis. For [B,C,N], move the middle channel axis last
    when it is a plausible feature dimension and the last dimension is a much
    larger token count.
    """
    value = np.asarray(array, dtype=np.float32)
    if value.ndim < 2:
        raise ValueError(f"feature must have >=2 dims, got {value.shape}")
    if value.ndim == 2:
        return value.reshape(-1, value.shape[-1]), int(value.shape[0])
    shape = value.shape
    # Candidate channel axes: typical 32..1024. Prefer last unless the layout is
    # clearly [B,C,N] with N larger than C.
    channel_axis = value.ndim - 1
    if value.ndim == 3:
        typical = {16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024}
        middle_typical = int(shape[1]) in typical
        last_typical = int(shape[2]) in typical
        if middle_typical and not last_typical:
            channel_axis = 1
        elif last_typical and not middle_typical:
            channel_axis = 2
        elif shape[2] > shape[1]:
            # Usual [B,C,N] layout with many query/angle tokens.
            channel_axis = 1
    value = np.moveaxis(value, channel_axis, -1)
    return value.reshape(-1, value.shape[-1]), int(np.prod(value.shape[:-1]))


def targets_for_token_count(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    gt: np.ndarray,
    token_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Align CDF target granularity to a feature token count."""
    base = student_logits.shape[:-1]
    total_entries = int(np.prod(base))
    if total_entries % token_count != 0:
        raise ValueError(f"CDF base shape {base} ({total_entries}) is not divisible by feature tokens {token_count}")
    output_groups = total_entries // token_count
    student_vec = student_logits.reshape(token_count, output_groups * student_logits.shape[-1])
    teacher_vec = teacher_logits.reshape(token_count, output_groups * teacher_logits.shape[-1])
    gt_vec = gt.reshape(token_count, output_groups * gt.shape[-1])
    granularity = f"token_to_{output_groups}xT"
    student_bce = F.binary_cross_entropy_with_logits(
        torch.from_numpy(student_vec), torch.from_numpy(gt_vec), reduction="none"
    ).mean(dim=1).numpy()
    teacher_bce = F.binary_cross_entropy_with_logits(
        torch.from_numpy(teacher_vec), torch.from_numpy(gt_vec), reduction="none"
    ).mean(dim=1).numpy()
    return gt_vec, sigmoid_np(teacher_vec), (teacher_bce < student_bce).astype(np.float32), granularity


def collect_layer_rows(args: argparse.Namespace) -> Dict[str, LayerRows]:
    aliases = load_aliases(args.mapping_json)
    requested = [token.strip() for token in args.layers.split(",") if token.strip()]
    files = discover_payload_files(args.diagnosis_dir)
    if int(args.max_files) > 0:
        files = files[: int(args.max_files)]
    if not files:
        raise FileNotFoundError(
            f"No NPZ/PT paired diagnostic files under {args.diagnosis_dir}"
        )
    try:
        progress_every = max(1, int(os.environ.get("PKD_P0_PROGRESS_EVERY", "50")))
    except ValueError:
        progress_every = 50
    print(
        f"[P0-D][LOAD] root={Path(args.diagnosis_dir).expanduser().resolve()} "
        f"files={len(files)} layers={requested} max_files={int(args.max_files)} "
        f"progress_every={progress_every}",
        flush=True,
    )
    print(
        "[P0-D][NOTE] max_rows_per_layer is applied after cache files are read; "
        "the loading phase is CPU/RAM/I/O bound and may leave the GPU idle.",
        flush=True,
    )
    buffers: Dict[str, Dict[str, List[np.ndarray]]] = {
        layer: {"feature": [], "gt": [], "teacher": [], "better": [], "scene": [], "teacher_feature": []}
        for layer in requested
    }
    granularity: Dict[str, str] = {}
    rejected: List[str] = []

    for file_index, path in enumerate(files, start=1):
        try:
            payload = load_payload(path)
            student_logits, teacher_logits, gt = cdf_tensors(payload, aliases)
            scene_id = scalar_scene_id(payload, aliases, path)
            for layer in requested:
                student_key = f"feature_{layer}_student"
                teacher_key = f"feature_{layer}_teacher"
                if student_key not in aliases:
                    continue
                _, student_feature_value = find_value(payload, aliases[student_key])
                if student_feature_value is None:
                    continue
                feature, token_count = feature_matrix(as_numpy(student_feature_value))
                gt_target, teacher_target, better, current_granularity = targets_for_token_count(
                    student_logits, teacher_logits, gt, token_count
                )
                if len(feature) != len(gt_target):
                    raise AssertionError("feature/target row alignment failed")
                _, teacher_feature_value = find_value(payload, aliases.get(teacher_key, ()))
                teacher_feature = None
                if teacher_feature_value is not None:
                    candidate, teacher_tokens = feature_matrix(as_numpy(teacher_feature_value))
                    if teacher_tokens == token_count:
                        teacher_feature = candidate
                buffers[layer]["feature"].append(feature)
                buffers[layer]["gt"].append(gt_target)
                buffers[layer]["teacher"].append(teacher_target)
                buffers[layer]["better"].append(better)
                buffers[layer]["scene"].append(np.full(token_count, scene_id, dtype=np.int16))
                if teacher_feature is not None:
                    buffers[layer]["teacher_feature"].append(teacher_feature)
                granularity[layer] = current_granularity
        except Exception as exc:
            rejected.append(f"{path}: {exc!r}")
        if file_index == 1 or file_index % progress_every == 0 or file_index == len(files):
            populated = {
                layer: len(data["feature"])
                for layer, data in buffers.items()
            }
            print(
                f"[P0-D][LOAD] {file_index}/{len(files)} rejected={len(rejected)} "
                f"feature_files={populated} file={path.name}",
                flush=True,
            )

    print(
        f"[P0-D][CONCAT] finished file scan; rejected={len(rejected)}. "
        "Concatenating and subsampling each requested layer.",
        flush=True,
    )
    rng = np.random.default_rng(int(args.seed))
    result: Dict[str, LayerRows] = {}
    for layer, data in buffers.items():
        if not data["feature"]:
            continue
        feature = np.concatenate(data["feature"], axis=0)
        gt_target = np.concatenate(data["gt"], axis=0)
        teacher_target = np.concatenate(data["teacher"], axis=0)
        better = np.concatenate(data["better"], axis=0)
        scene = np.concatenate(data["scene"], axis=0)
        teacher_feature = None
        if len(data["teacher_feature"]) == len(data["feature"]):
            teacher_feature = np.concatenate(data["teacher_feature"], axis=0)
        maximum = int(args.max_rows_per_layer)
        if maximum > 0 and len(feature) > maximum:
            # Stratify by train/validation membership before subsampling.
            train_ids = np.flatnonzero(scene < int(args.val_scene_start))
            val_ids = np.flatnonzero(scene >= int(args.val_scene_start))
            train_n = min(len(train_ids), int(maximum * 0.8))
            val_n = min(len(val_ids), maximum - train_n)
            ids = np.concatenate([
                rng.choice(train_ids, train_n, replace=False) if train_n else np.empty(0, dtype=np.int64),
                rng.choice(val_ids, val_n, replace=False) if val_n else np.empty(0, dtype=np.int64),
            ])
            rng.shuffle(ids)
            feature, gt_target, teacher_target, better, scene = (
                array[ids] for array in (feature, gt_target, teacher_target, better, scene)
            )
            if teacher_feature is not None:
                teacher_feature = teacher_feature[ids]
        result[layer] = LayerRows(
            layer=layer,
            feature=feature.astype(np.float32),
            gt_target=gt_target.astype(np.float32),
            teacher_target=teacher_target.astype(np.float32),
            teacher_better=better.astype(np.float32),
            scene_id=scene.astype(np.int16),
            teacher_feature=None if teacher_feature is None else teacher_feature.astype(np.float32),
            granularity=granularity[layer],
        )
    if not result:
        raise RuntimeError("No requested layer could be aligned. Rejected examples:\n" + "\n".join(rejected[:30]))
    return result


class Probe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kind: str, hidden_dim: int) -> None:
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(input_dim, output_dim)
        elif kind == "mlp":
            hidden = min(max(32, hidden_dim), max(32, input_dim * 2))
            self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, output_dim))
        else:
            raise ValueError(kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_loaders(
    features: np.ndarray,
    targets: np.ndarray,
    scene_ids: np.ndarray,
    val_scene_start: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    train_mask = scene_ids < val_scene_start
    val_mask = scene_ids >= val_scene_start
    if not train_mask.any() or not val_mask.any():
        raise RuntimeError(
            f"Need both train and validation rows around val_scene_start={val_scene_start}; "
            f"scene range={scene_ids.min()}..{scene_ids.max()}"
        )
    mean = features[train_mask].mean(axis=0, keepdims=True)
    std = features[train_mask].std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-5)
    normalized = (features - mean) / std

    def loader(mask: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(normalized[mask]).float(),
            torch.from_numpy(targets[mask]).float(),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=max(0, num_workers),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    return loader(train_mask, True), loader(val_mask, False), mean.squeeze(0), std.squeeze(0)


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Tie-corrected average ranks.
    values, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for group, count in enumerate(counts):
        if count > 1:
            ids = np.flatnonzero(inverse == group)
            ranks[ids] = ranks[ids].mean()
    rank_sum = ranks[labels].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def train_probe(
    layer_rows: LayerRows,
    task: str,
    kind: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    if task == "gt_cdf":
        targets = layer_rows.gt_target
        output_dim = targets.shape[1]
        loss_type = "bce"
    elif task == "teacher_cdf":
        targets = layer_rows.teacher_target
        output_dim = targets.shape[1]
        loss_type = "bce"
    elif task == "teacher_better":
        targets = layer_rows.teacher_better[:, None]
        output_dim = 1
        loss_type = "bce"
    else:
        raise ValueError(task)

    train_loader, val_loader, mean, std = make_loaders(
        layer_rows.feature,
        targets,
        layer_rows.scene_id,
        int(args.val_scene_start),
        int(args.batch_size),
        int(args.num_workers),
    )
    model = Probe(layer_rows.feature.shape[1], output_dim, kind, int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    best_state = None
    best_val = float("inf")

    for epoch_index in range(int(args.epochs)):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        total, elements = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                loss = F.binary_cross_entropy_with_logits(model(x), y, reduction="sum")
                total += float(loss.item())
                elements += y.numel()
        val = total / max(elements, 1)
        if val < best_val:
            best_val = val
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"[P0-D][EPOCH] layer={layer_rows.layer} probe={kind} task={task} "
            f"epoch={epoch_index + 1}/{int(args.epochs)} val_bce={val:.6f} "
            f"best={best_val:.6f}",
            flush=True,
        )
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    predictions, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x.to(device, non_blocking=True)).cpu()
            predictions.append(torch.sigmoid(logits).numpy())
            labels.append(y.numpy())
    pred = np.concatenate(predictions, axis=0)
    true = np.concatenate(labels, axis=0)
    result: Dict[str, Any] = {
        "layer": layer_rows.layer,
        "granularity": layer_rows.granularity,
        "task": task,
        "probe": kind,
        "input_dim": int(layer_rows.feature.shape[1]),
        "output_dim": int(output_dim),
        "train_rows": int((layer_rows.scene_id < int(args.val_scene_start)).sum()),
        "val_rows": int((layer_rows.scene_id >= int(args.val_scene_start)).sum()),
        "val_bce": float(best_val),
        "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    if task in {"gt_cdf", "teacher_cdf"}:
        result["val_probability_mae"] = float(np.abs(pred - true).mean())
        result["val_utility_mae"] = float(np.abs(pred.mean(axis=1) - true.mean(axis=1)).mean())
    else:
        result["val_accuracy"] = float(((pred[:, 0] >= 0.5) == (true[:, 0] >= 0.5)).mean())
        result["val_auc"] = binary_auc(pred[:, 0], true[:, 0])
    return result


def teacher_feature_probe(
    layer_rows: LayerRows,
    kind: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    if layer_rows.teacher_feature is None:
        return None
    train_mask = layer_rows.scene_id < int(args.val_scene_start)
    val_mask = ~train_mask
    if not train_mask.any() or not val_mask.any():
        return None
    teacher_train = torch.from_numpy(layer_rows.teacher_feature[train_mask]).float()
    teacher_mean = teacher_train.mean(dim=0, keepdim=True)
    centered = teacher_train - teacher_mean
    q = min(int(args.pca_dim), centered.shape[1], max(1, centered.shape[0] - 1))
    _, _, basis = torch.pca_lowrank(centered, q=q, center=False)
    teacher_all = (torch.from_numpy(layer_rows.teacher_feature).float() - teacher_mean) @ basis
    targets = teacher_all.numpy().astype(np.float32)
    train_loader, val_loader, _, _ = make_loaders(
        layer_rows.feature, targets, layer_rows.scene_id, int(args.val_scene_start), int(args.batch_size), int(args.num_workers)
    )
    model = Probe(layer_rows.feature.shape[1], q, kind, int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    best_state, best_val = None, float("inf")
    for epoch_index in range(int(args.epochs)):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            loss = F.smooth_l1_loss(prediction, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        total, elements = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                loss = F.smooth_l1_loss(model(x), y, reduction="sum")
                total += float(loss.item())
                elements += y.numel()
        value = total / max(elements, 1)
        if value < best_val:
            best_val = value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"[P0-D][EPOCH] layer={layer_rows.layer} probe={kind} "
            f"task=teacher_feature_pca epoch={epoch_index + 1}/{int(args.epochs)} "
            f"val_smooth_l1={value:.6f} best={best_val:.6f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    predictions, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            predictions.append(model(x.to(device)).cpu())
            labels.append(y)
    prediction = torch.cat(predictions)
    label = torch.cat(labels)
    cosine = F.cosine_similarity(prediction, label, dim=1).mean()
    variance = torch.sum((label - label.mean(dim=0, keepdim=True)) ** 2)
    residual = torch.sum((prediction - label) ** 2)
    return {
        "layer": layer_rows.layer,
        "granularity": layer_rows.granularity,
        "task": "teacher_feature_pca",
        "probe": kind,
        "input_dim": int(layer_rows.feature.shape[1]),
        "output_dim": int(q),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "val_smooth_l1": float(best_val),
        "val_cosine": float(cosine.item()),
        "val_r2": float((1.0 - residual / (variance + 1e-12)).item()),
        "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(args.seed))
    device = torch.device(args.device)
    print(
        f"[P0-D][START] diagnosis_dir={Path(args.diagnosis_dir).expanduser().resolve()} "
        f"output_dir={output_dir} device={device} epochs={int(args.epochs)} "
        f"max_files={int(args.max_files)} max_rows_per_layer={int(args.max_rows_per_layer)}",
        flush=True,
    )
    layers = collect_layer_rows(args)
    results: List[Dict[str, Any]] = []
    for layer_name, rows in layers.items():
        print(
            f"[P0-D] layer={layer_name} rows={len(rows.feature)} input={rows.feature.shape[1]} granularity={rows.granularity}",
            flush=True,
        )
        for kind in ("linear", "mlp"):
            for task in ("gt_cdf", "teacher_cdf", "teacher_better"):
                print(
                    f"[P0-D][PROBE-START] layer={layer_name} probe={kind} task={task}",
                    flush=True,
                )
                result = train_probe(rows, task, kind, args, device)
                results.append(result)
                print(
                    f"[P0-D] {layer_name}/{kind}/{task}: val_bce={result['val_bce']:.6f}",
                    flush=True,
                )
            print(
                f"[P0-D][PROBE-START] layer={layer_name} probe={kind} "
                "task=teacher_feature_pca",
                flush=True,
            )
            feature_result = teacher_feature_probe(rows, kind, args, device)
            if feature_result is not None:
                results.append(feature_result)
                print(
                    f"[P0-D] {layer_name}/{kind}/teacher_feature_pca: cosine={feature_result['val_cosine']:.4f}",
                    flush=True,
                )

    write_csv(output_dir / "probe_results.csv", results)
    atomic_json_dump(
        {
            "experiment": "P0-D layer-wise decodability",
            "diagnosis_dir": str(Path(args.diagnosis_dir).expanduser().resolve()),
            "val_scene_start": int(args.val_scene_start),
            "layers": {
                name: {
                    "rows": int(len(rows.feature)),
                    "input_dim": int(rows.feature.shape[1]),
                    "target_dim": int(rows.gt_target.shape[1]),
                    "granularity": rows.granularity,
                    "teacher_feature_available": rows.teacher_feature is not None,
                }
                for name, rows in layers.items()
            },
            "results": results,
        },
        output_dir / "summary.json",
    )
    print(f"[DONE] P0-D outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
