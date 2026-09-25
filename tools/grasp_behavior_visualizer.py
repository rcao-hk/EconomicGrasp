"""Reusable visualization utilities for EconomicGrasp behavior inspection.

The module is deliberately experiment-agnostic. It consumes already-computed
RGB/depth/features/actions/logits/debug tensors and writes deterministic PNG/NPZ
artifacts. Training/inference scripts can gate calls with BehaviorVisConfig.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple
import json
import math

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ITEMS = (
    "rgb",
    "depth",
    "pointcloud",
    "features",
    "proposal",
    "stage1",
    "dcr",
    "cdf",
    "local",
    "grasp_delta",
    "evaluation",
    "air",
)


@dataclass
class BehaviorVisConfig:
    output_root: str
    items: Sequence[str] = field(default_factory=lambda: DEFAULT_ITEMS)
    every: int = 0
    start: int = 0
    end: int = -1
    topk: int = 20
    max_points: int = 50000
    dpi: int = 140
    save_npz: bool = True

    @classmethod
    def from_strings(cls, output_root: str, items: str = "all", **kwargs):
        parsed = DEFAULT_ITEMS if items.strip().lower() == "all" else tuple(
            x.strip() for x in items.split(",") if x.strip())
        bad = sorted(set(parsed) - set(DEFAULT_ITEMS))
        if bad:
            raise ValueError(f"Unknown visualization items: {bad}")
        return cls(output_root=output_root, items=parsed, **kwargs)


class BehaviorVisualizer:
    """Small gate/helper that can be called inside arbitrary experiments."""

    def __init__(self, config: BehaviorVisConfig):
        self.cfg = config
        self.root = Path(config.output_root)
        self.enabled: Set[str] = set(config.items)

    def should_save(self, iteration: int) -> bool:
        it = int(iteration)
        if it < self.cfg.start:
            return False
        if self.cfg.end >= 0 and it > self.cfg.end:
            return False
        return self.cfg.every > 0 and (it - self.cfg.start) % self.cfg.every == 0

    def wants(self, item: str) -> bool:
        return item in self.enabled

    def frame_dir(self, split: str, scene_id: int, anno_id: int,
                  case: str = "nominal", iteration: Optional[int] = None) -> Path:
        stem = self.root / split / f"scene_{int(scene_id):04d}" / f"ann_{int(anno_id):04d}"
        if iteration is not None:
            stem = stem / f"iter_{int(iteration):07d}"
        out = stem / sanitize_name(case)
        out.mkdir(parents=True, exist_ok=True)
        return out

    def write_manifest(self, path: Path, payload: Mapping):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


def sanitize_name(text: str) -> str:
    return str(text).replace(":", "_").replace("+", "p").replace("-", "m").replace(".", "p")


def _jsonable(x):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, Path):
        return str(x)
    if torch.is_tensor(x):
        if x.numel() == 1:
            return float(x.detach().cpu())
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.reshape(-1)[0])
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return str(x)


def to_numpy(x, dtype=None):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return x.astype(dtype, copy=False) if dtype is not None else x


def denormalize_rgb(img) -> np.ndarray:
    """ImageNet-normalized CHW/BCHW -> HWC in [0,1]."""
    x = to_numpy(img, np.float32)
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected CHW image, got {x.shape}")
    if x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    return np.clip(x * std + mean, 0., 1.)


def robust_limits(x, low=1., high=99.):
    a = to_numpy(x, np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return 0., 1.
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def normalize_map(x, low=1., high=99.):
    a = to_numpy(x, np.float32)
    lo, hi = robust_limits(a, low, high)
    return np.clip((a - lo) / (hi - lo), 0., 1.)


def feature_norm_map(feature) -> np.ndarray:
    f = to_numpy(feature, np.float32)
    if f.ndim == 4:
        f = f[0]
    if f.ndim != 3:
        raise ValueError(f"Expected CHW feature, got {f.shape}")
    return np.linalg.norm(f, axis=0)


def feature_pca_rgb(feature, max_samples=30000) -> np.ndarray:
    """PCA-to-RGB diagnostic; sign is deterministic by largest-loading entry."""
    f = to_numpy(feature, np.float32)
    if f.ndim == 4:
        f = f[0]
    if f.ndim != 3:
        raise ValueError(f"Expected CHW feature, got {f.shape}")
    c, h, w = f.shape
    x = f.reshape(c, -1).T
    x = x - x.mean(0, keepdims=True)
    if len(x) > max_samples:
        idx = np.linspace(0, len(x) - 1, max_samples).round().astype(np.int64)
        fit = x[idx]
    else:
        fit = x
    try:
        _, _, vt = np.linalg.svd(fit, full_matrices=False)
        basis = vt[:3].T
        for j in range(basis.shape[1]):
            k = np.argmax(np.abs(basis[:, j]))
            if basis[k, j] < 0:
                basis[:, j] *= -1
        y = x @ basis
    except np.linalg.LinAlgError:
        y = x[:, :min(3, c)]
        if y.shape[1] < 3:
            y = np.pad(y, ((0, 0), (0, 3 - y.shape[1])))
    y = y.reshape(h, w, 3)
    out = np.empty_like(y)
    for j in range(3):
        out[..., j] = normalize_map(y[..., j])
    return out


def backproject_depth(depth, K, stride=2, valid_range=(.05, 2.0)):
    d = to_numpy(depth, np.float32)
    if d.ndim == 4:
        d = d[0, 0]
    elif d.ndim == 3:
        d = d[0]
    K = to_numpy(K, np.float32)
    if K.ndim == 3:
        K = K[0]
    h, w = d.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = d[::stride, ::stride]
    ok = np.isfinite(z) & (z > valid_range[0]) & (z < valid_range[1])
    x = (xs - K[0, 2]) / K[0, 0] * z
    y = (ys - K[1, 2]) / K[1, 1] * z
    pts = np.stack((x, y, z), -1)[ok]
    pix = np.stack((xs, ys), -1)[ok]
    return pts.astype(np.float32), pix.astype(np.float32)


def project_xyz(xyz, K):
    p = to_numpy(xyz, np.float32)
    K = to_numpy(K, np.float32)
    if K.ndim == 3:
        K = K[0]
    z = p[..., 2]
    safe = np.maximum(z, 1e-8)
    u = K[0, 0] * p[..., 0] / safe + K[0, 2]
    v = K[1, 1] * p[..., 1] / safe + K[1, 2]
    return np.stack((u, v), -1), z > 1e-8


def grasp_skeleton_points(actions, finger_width=.01, finger_length=.06, approach_dist=.03):
    """Return [N,7,3] sparse gripper skeleton points for GraspNet rows."""
    a = to_numpy(actions, np.float32)
    if a.ndim == 1:
        a = a[None]
    width, height, depth = a[:, 1], a[:, 2], a[:, 3]
    R = a[:, 4:13].reshape(-1, 3, 3)
    t = a[:, 13:16]
    n = len(a)
    local = np.zeros((n, 7, 3), np.float32)
    local[:, 0, 0] = depth - finger_length - finger_width - approach_dist
    local[:, 1, 0] = depth - finger_length
    local[:, 2, 0] = depth
    local[:, 3, 0] = depth
    local[:, 4, 0] = depth - finger_length
    local[:, 5, 0] = depth - finger_length
    local[:, 6, 0] = 0.
    local[:, 1, 1] = -width / 2
    local[:, 2, 1] = -width / 2
    local[:, 3, 1] = width / 2
    local[:, 4, 1] = width / 2
    # point 5/6 are axis hints
    local[:, 5, 2] = np.maximum(height / 2, .01)
    local[:, 6, 1] = np.maximum(width / 2, .02)
    return t[:, None, :] + np.matmul(local, np.swapaxes(R, 1, 2))


def save_rgb(path: Path, rgb, title="RGB"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_depth_panel(path: Path, nominal, active, title="", sensor=None):
    n = to_numpy(nominal, np.float32).squeeze()
    a = to_numpy(active, np.float32).squeeze()
    delta = (a - n) * 1000.
    panels = [(n, "predicted depth [m]"), (a, "active depth [m]"), (delta, "active - predicted [mm]")]
    if sensor is not None:
        panels.insert(1, (to_numpy(sensor, np.float32).squeeze(), "sensor depth [m] (context only)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    axes = np.atleast_1d(axes)
    for ax, (arr, name) in zip(axes, panels):
        if "mm]" in name:
            lim = max(abs(robust_limits(arr, 1, 99)[0]), abs(robust_limits(arr, 1, 99)[1]), 1e-6)
            im = ax.imshow(arr, vmin=-lim, vmax=lim, cmap="coolwarm")
        else:
            lo, hi = robust_limits(arr[arr > 0] if np.any(arr > 0) else arr)
            im = ax.imshow(arr, vmin=lo, vmax=hi, cmap="viridis")
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_feature_panel(path: Path, raw, enhanced, title="Feature behavior"):
    raw_norm = feature_norm_map(raw)
    enh_norm = feature_norm_map(enhanced)
    raw_pca = feature_pca_rgb(raw)
    enh_pca = feature_pca_rgb(enhanced)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, arr, name in (
        (axes[0, 0], raw_pca, "pre-enhancer PCA"),
        (axes[0, 1], enh_pca, "post-enhancer PCA"),
        (axes[1, 0], raw_norm, "pre-enhancer feature norm"),
        (axes[1, 1], enh_norm, "post-enhancer feature norm"),
    ):
        im = ax.imshow(arr, cmap=None if arr.ndim == 3 else "magma")
        ax.set_title(name)
        ax.axis("off")
        if arr.ndim == 2:
            fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_proposal_panel(path: Path, proposal_logits, rgb=None, title="Proposal maps"):
    p = to_numpy(proposal_logits, np.float32)
    if p.ndim == 4:
        p = p[0]
    fg = np.exp(p[:2] - p[:2].max(0, keepdims=True))
    fg = fg[1] / np.maximum(fg.sum(0), 1e-8)
    grasp = p[2]
    grasp_prob = 1. / (1. + np.exp(-grasp))
    panels = [(fg, "objectness foreground"), (grasp_prob, "graspness")]
    if rgb is not None:
        panels.insert(0, (rgb, "RGB"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    axes = np.atleast_1d(axes)
    for ax, (arr, name) in zip(axes, panels):
        im = ax.imshow(arr, cmap=None if np.asarray(arr).ndim == 3 else "viridis")
        ax.set_title(name)
        ax.axis("off")
        if np.asarray(arr).ndim == 2:
            fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_pointcloud_views(path: Path, point_sets: Mapping[str, np.ndarray],
                          colors: Optional[np.ndarray] = None,
                          max_points=40000, title="Geometry"):
    names = list(point_sets)
    fig = plt.figure(figsize=(6 * len(names), 6))
    for i, name in enumerate(names, 1):
        pts = np.asarray(point_sets[name], np.float32)
        if len(pts) > max_points:
            idx = np.linspace(0, len(pts) - 1, max_points).round().astype(np.int64)
            pts = pts[idx]
        ax = fig.add_subplot(1, len(names), i, projection="3d")
        c = pts[:, 2] if len(pts) else None
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, s=.4, cmap="viridis", alpha=.8)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=25, azim=-70)
        if len(pts):
            center = np.median(pts, axis=0)
            radius = np.percentile(np.linalg.norm(pts - center, axis=1), 95)
            radius = max(float(radius), .05)
            ax.set_xlim(center[0]-radius, center[0]+radius)
            ax.set_ylim(center[1]-radius, center[1]+radius)
            ax.set_zlim(center[2]-radius, center[2]+radius)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_grasp_overlay(path: Path, rgb, K, groups: Mapping[str, np.ndarray],
                       topk=20, title="Grasp overlay"):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb)
    line_styles = ["-", "--", ":", "-."]
    for gi, (name, actions) in enumerate(groups.items()):
        a = np.asarray(actions, np.float32)
        if not len(a):
            continue
        order = np.argsort(-a[:, 0], kind="stable")[:min(topk, len(a))]
        skel = grasp_skeleton_points(a[order])
        uv, valid = project_xyz(skel, K)
        for row, ok in zip(uv, valid):
            if not np.all(ok[:5]):
                continue
            # approach -> left root -> left tip; then right tip -> right root
            ax.plot(row[[0, 1, 2], 0], row[[0, 1, 2], 1],
                    line_styles[gi % len(line_styles)], linewidth=.8, alpha=.65)
            ax.plot(row[[3, 4], 0], row[[3, 4], 1],
                    line_styles[gi % len(line_styles)], linewidth=.8, alpha=.65)
        centers = uv[:, 0]
        ax.scatter(centers[:, 0], centers[:, 1], s=8, alpha=.7, label=name)
    ax.set_title(title)
    ax.axis("off")
    if groups:
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_center_shift_overlay(path: Path, rgb, K, before, after,
                              scores=None, topk=50, title="Center shifts"):
    b = np.asarray(before, np.float32)
    a = np.asarray(after, np.float32)
    if b.shape != a.shape or b.shape[-1] != 17:
        raise ValueError("before/after grasp arrays must align")
    if scores is None:
        scores = b[:, 0]
    order = np.argsort(-np.asarray(scores), kind="stable")[:min(topk, len(b))]
    buv, bv = project_xyz(b[order, 13:16], K)
    auv, av = project_xyz(a[order, 13:16], K)
    good = bv & av
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb)
    for p0, p1 in zip(buv[good], auv[good]):
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="->", linewidth=.7, alpha=.65))
    ax.scatter(buv[good, 0], buv[good, 1], s=10, label="before", alpha=.75)
    ax.scatter(auv[good, 0], auv[good, 1], s=10, label="after", alpha=.75)
    ax.set_title(title)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_cdf_heatmap(path: Path, logits, offsets_mm, query_scores,
                     max_queries=32, title="CDF / center utility"):
    l = to_numpy(logits, np.float32)
    offsets = to_numpy(offsets_mm, np.float32).reshape(-1)
    qs = to_numpy(query_scores, np.float32).reshape(-1)
    qidx = np.argsort(-qs, kind="stable")[:min(max_queries, len(qs))]
    util = 1. / (1. + np.exp(-l))
    util = util.mean(-1)[:, qidx].T
    fig, ax = plt.subplots(figsize=(max(7, len(offsets) * .8), max(5, len(qidx) * .22)))
    im = ax.imshow(util, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(offsets)))
    ax.set_xticklabels([f"{x:g}" for x in offsets])
    ax.set_xlabel("camera-z offset [mm]")
    ax.set_ylabel("query (Stage-1 score order)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="mean CDF utility")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_offset_response(path: Path, rgb, K, bundle, selected,
                         offsets_mm, query_scores, max_queries=100,
                         title="Selected center offsets"):
    actions = to_numpy(bundle["actions"], np.float32)
    sel = to_numpy(selected, np.int64).reshape(-1)
    offsets = to_numpy(offsets_mm, np.float32).reshape(-1)
    score = to_numpy(query_scores, np.float32).reshape(-1)
    order = np.argsort(-score, kind="stable")[:min(max_queries, len(score))]
    native = actions[np.flatnonzero(np.isclose(offsets, 0.))[0], order, 13:16]
    uv, valid = project_xyz(native, K)
    values = offsets[sel[order]]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb)
    sc = ax.scatter(uv[valid, 0], uv[valid, 1], c=values[valid], s=20,
                    cmap="coolwarm", vmin=-max(abs(offsets)), vmax=max(abs(offsets)))
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(sc, ax=ax, label="selected offset [mm]")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_local_patch_overlay(path: Path, rgb, patch_uv, patch_attn=None,
                             center_uv=None, max_queries=12,
                             title="CVA local analysis regions"):
    p = to_numpy(patch_uv, np.float32)
    if p.ndim == 4:
        p = p[0]
    p = p[:max_queries]
    attn = None if patch_attn is None else to_numpy(patch_attn, np.float32)
    if attn is not None and attn.ndim == 3:
        attn = attn[:max_queries]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb)
    for qi, pts in enumerate(p):
        if not np.isfinite(pts).all():
            continue
        if attn is None:
            size = np.full(len(pts), 6.)
        else:
            aa = attn[qi].reshape(-1)
            size = 5. + 60. * aa / max(float(aa.max()), 1e-8)
        ax.scatter(pts[:, 0], pts[:, 1], s=size, alpha=.45)
        if center_uv is not None:
            c = to_numpy(center_uv, np.float32)
            if c.ndim == 3:
                c = c[0]
            if qi < len(c):
                ax.text(c[qi, 0], c[qi, 1], str(qi), fontsize=7)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_rank_residual_panel(path: Path, residual, offsets_mm, query_scores,
                             max_queries=32, title="DCR ranking residual"):
    r = to_numpy(residual, np.float32)
    offsets = to_numpy(offsets_mm, np.float32).reshape(-1)
    score = to_numpy(query_scores, np.float32).reshape(-1)
    qidx = np.argsort(-score, kind="stable")[:min(max_queries, len(score))]
    data = r[:, qidx].T
    lim = max(np.percentile(np.abs(data), 99), 1e-6)
    fig, ax = plt.subplots(figsize=(max(7, len(offsets) * .8), max(5, len(qidx) * .22)))
    im = ax.imshow(data, aspect="auto", vmin=-lim, vmax=lim, cmap="coolwarm")
    ax.set_xticks(np.arange(len(offsets)))
    ax.set_xticklabels([f"{x:g}" for x in offsets])
    ax.set_xlabel("camera-z offset [mm]")
    ax.set_ylabel("query (Stage-1 score order)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="log-odds residual")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_evaluation_overlay(path: Path, rgb, K, grasps, accuracy,
                            topk=50, title="Evaluator-colored grasps"):
    a = np.asarray(grasps, np.float32)
    acc = np.asarray(accuracy, np.float32)
    if acc.ndim == 2:
        quality = acc.mean(-1)
    elif acc.ndim == 1:
        quality = acc
    else:
        raise ValueError(f"Expected rank x threshold accuracy, got {acc.shape}")
    n = min(len(a), len(quality), topk)
    uv, valid = project_xyz(a[:n, 13:16], K)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb)
    sc = ax.scatter(uv[:n][valid[:n], 0], uv[:n][valid[:n], 1],
                    c=quality[:n][valid[:n]], s=28, vmin=0, vmax=1,
                    cmap="viridis")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(sc, ax=ax, label="mean evaluator success across friction thresholds")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_ply(path: Path, points, rgb=None):
    pts = np.asarray(points, np.float32)
    if rgb is None:
        col = np.full((len(pts), 3), 180, np.uint8)
    else:
        col = np.asarray(rgb)
        if col.max(initial=0) <= 1:
            col = np.clip(col * 255, 0, 255)
        col = col.astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, col):
            f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
