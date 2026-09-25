"""Reusable visual diagnostics for EconomicGrasp-DPT/CVA/CDF experiments.

The functions in this module are deliberately model-agnostic: experiment
runners pass tensors/end-points/grasp arrays in, and the module writes PNG/PLY/
CSV/JSON artifacts.  Lightweight hooks can call the same helpers every N
iterations; expensive GraspNet/Dex-Net evaluation stays in offline runners.
"""
from __future__ import annotations

import csv
import html
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], np.float32)

ITEM_PRESETS = {
    "light": (
        "rgb,depth,proposal,feature,query_response,grasps"
    ),
    "core": (
        "rgb,depth,pointcloud,proposal,feature,spatial,view,cdf,"
        "query_response,local,grasps,corruption_delta"
    ),
    "all": (
        "rgb,depth,pointcloud,proposal,feature,spatial,view,cdf,"
        "query_response,local,grasps,corruption_delta,evaluator,air"
    ),
}


def parse_items(text: str) -> set[str]:
    value = str(text or "core").strip().lower()
    if value in ITEM_PRESETS:
        value = ITEM_PRESETS[value]
    items = {x.strip() for x in value.split(",") if x.strip()}
    if "all" in items:
        items = set(ITEM_PRESETS["all"].split(","))
    return items


def should_save(iteration: Optional[int], every: int) -> bool:
    if every <= 0:
        return False
    if iteration is None:
        return True
    return int(iteration) % int(every) == 0


class BehaviorVizWriter:
    """Small scheduling/output wrapper reusable from training or inference."""

    def __init__(self, root: str | Path, items: str = "core",
                 every: int = 1, dpi: int = 160):
        self.root = Path(root)
        self.items = parse_items(items)
        self.every = int(every)
        self.dpi = int(dpi)

    def enabled(self, item: str) -> bool:
        return str(item) in self.items

    def due(self, iteration: Optional[int]) -> bool:
        return should_save(iteration, self.every)

    def frame_dir(self, split: str, scene_id: int, anno_id: int,
                  case: str = "nominal", iteration: Optional[int] = None) -> Path:
        case_key = str(case).replace(":", "_").replace("+", "p").replace("-", "m").replace(".", "p")
        out = self.root / split / f"scene_{int(scene_id):04d}" / f"ann_{int(anno_id):04d}" / case_key
        if iteration is not None:
            out = out / f"iter_{int(iteration):07d}"
        out.mkdir(parents=True, exist_ok=True)
        return out


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    return np.nan_to_num(np.asarray(x), nan=0.0, posinf=0.0, neginf=0.0)


def robust_limits(x: Any, low: float = 1.0, high: float = 99.0) -> Tuple[float, float]:
    a = to_numpy(x)
    finite = np.isfinite(a)
    if not finite.any():
        return 0.0, 1.0
    lo = float(np.percentile(a[finite], low))
    hi = float(np.percentile(a[finite], high))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def imagenet_rgb(img: Any) -> np.ndarray:
    """Convert normalized BCHW/CHW RGB to display HWC [0,1]."""
    x = to_numpy(img)
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected RGB CHW/BCHW, got {x.shape}")
    if x.shape[0] > 3:
        x = x[:3]
    if x.shape[0] == 1:
        x = np.repeat(x, 3, axis=0)
    x = x.transpose(1, 2, 0)
    x = x * IMAGENET_STD.reshape(1, 1, 3) + IMAGENET_MEAN.reshape(1, 1, 3)
    return np.clip(x, 0.0, 1.0)


def save_rgb(path: str | Path, rgb: Any, title: str = "") -> None:
    arr = to_numpy(rgb)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.imshow(np.clip(arr, 0, 1))
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout(pad=0.1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_heatmap(path: str | Path, value: Any, title: str = "",
                 cmap: str = "viridis", vmin=None, vmax=None,
                 colorbar: bool = True) -> None:
    arr = np.squeeze(to_numpy(value))
    if arr.ndim != 2:
        raise ValueError(f"Heatmap must be 2D, got {arr.shape}")
    if vmin is None or vmax is None:
        lo, hi = robust_limits(arr)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_overlay(path: str | Path, rgb: np.ndarray, value: Any,
                 title: str = "", cmap: str = "magma", alpha: float = .45,
                 vmin=None, vmax=None) -> None:
    arr = np.squeeze(to_numpy(value))
    h, w = rgb.shape[:2]
    if arr.shape != (h, w):
        t = torch.as_tensor(arr).float()[None, None]
        arr = F.interpolate(t, size=(h, w), mode="bilinear",
                            align_corners=False)[0, 0].numpy()
    if vmin is None or vmax is None:
        lo, hi = robust_limits(arr)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.imshow(np.clip(rgb, 0, 1))
    im = ax.imshow(arr, cmap=cmap, alpha=float(alpha), vmin=vmin, vmax=vmax)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def feature_pca_rgb(feature: Any, max_fit_points: int = 4096) -> np.ndarray:
    x = torch.as_tensor(feature).detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected CxHxW feature, got {tuple(x.shape)}")
    c, h, w = x.shape
    mat = x.permute(1, 2, 0).reshape(-1, c)
    mat = torch.nan_to_num(mat)
    mat = mat - mat.mean(0, keepdim=True)
    fit = mat
    if len(fit) > max_fit_points:
        ids = torch.linspace(0, len(fit) - 1, max_fit_points).long()
        fit = fit[ids]
    try:
        _, _, vh = torch.linalg.svd(fit, full_matrices=False)
        comp = vh[:min(3, vh.shape[0])].T
        out = mat @ comp
    except Exception:
        out = mat[:, :min(3, c)]
    if out.shape[1] < 3:
        out = torch.cat((out, torch.zeros(len(out), 3 - out.shape[1])), 1)
    out = out.reshape(h, w, 3).numpy()
    for j in range(3):
        lo, hi = robust_limits(out[..., j])
        out[..., j] = (out[..., j] - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def save_feature_bundle(out_dir: str | Path, raw: Any, enhanced: Any) -> None:
    out = Path(out_dir)
    raw_rgb = feature_pca_rgb(raw)
    enh_rgb = feature_pca_rgb(enhanced)
    save_rgb(out / "feature_pre_enhancer_pca.png", raw_rgb, "Pre-enhancer feature PCA")
    save_rgb(out / "feature_post_enhancer_pca.png", enh_rgb, "Post-enhancer feature PCA")
    raw_norm = torch.as_tensor(raw).float().norm(dim=1 if torch.as_tensor(raw).ndim == 4 else 0)
    enh_norm = torch.as_tensor(enhanced).float().norm(dim=1 if torch.as_tensor(enhanced).ndim == 4 else 0)
    save_heatmap(out / "feature_pre_norm.png", raw_norm, "Pre-enhancer feature norm")
    save_heatmap(out / "feature_post_norm.png", enh_norm, "Post-enhancer feature norm")


def depth_to_points(depth: Any, K: Any, rgb: Optional[np.ndarray] = None,
                    stride: int = 2, min_depth: float = .05,
                    max_depth: float = 2.0):
    d = np.squeeze(to_numpy(depth)).astype(np.float32)
    k = to_numpy(K)
    if k.ndim == 3:
        k = k[0]
    h, w = d.shape
    ys, xs = np.mgrid[0:h:max(1, stride), 0:w:max(1, stride)]
    z = d[::max(1, stride), ::max(1, stride)]
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    x = (xs - float(k[0, 2])) / float(k[0, 0]) * z
    y = (ys - float(k[1, 2])) / float(k[1, 1]) * z
    points = np.stack((x, y, z), -1)[valid].astype(np.float32)
    colors = None
    if rgb is not None:
        img = np.asarray(rgb)
        if img.shape[:2] != (h, w):
            import cv2
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        colors = np.clip(img[::max(1, stride), ::max(1, stride)][valid], 0, 1)
    return points, colors


def write_points_ply(path: str | Path, points: Any,
                     colors: Optional[Any] = None) -> None:
    p = np.asarray(points, np.float32).reshape(-1, 3)
    c = None
    if colors is not None:
        c = np.asarray(colors).reshape(-1, 3)
        if len(c) != len(p):
            raise ValueError("points/colors length mismatch")
        if np.issubdtype(c.dtype, np.floating):
            c = np.clip(c * 255.0, 0, 255).astype(np.uint8)
        else:
            c = np.clip(c, 0, 255).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(p)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if c is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if c is None:
            for xyz in p:
                f.write(f"{xyz[0]:.7g} {xyz[1]:.7g} {xyz[2]:.7g}\n")
        else:
            for xyz, rgbv in zip(p, c):
                f.write(f"{xyz[0]:.7g} {xyz[1]:.7g} {xyz[2]:.7g} "
                        f"{int(rgbv[0])} {int(rgbv[1])} {int(rgbv[2])}\n")


def save_depth_bundle(out_dir: str | Path, rgb: np.ndarray, K: Any,
                      nominal: Any, active: Any,
                      sensor: Optional[Any] = None,
                      rendered: Optional[Any] = None,
                      point_stride: int = 3) -> None:
    out = Path(out_dir)
    nom = np.squeeze(to_numpy(nominal))
    act = np.squeeze(to_numpy(active))
    delta_mm = (act - nom) * 1000.0
    save_heatmap(out / "depth_pred_nominal.png", nom, "Predicted metric depth", vmin=.2, vmax=1.0)
    save_heatmap(out / "depth_active.png", act, "Active/corrupted predicted depth", vmin=.2, vmax=1.0)
    lim = max(1.0, float(np.percentile(np.abs(delta_mm), 99)))
    save_heatmap(out / "depth_corruption_delta_mm.png", delta_mm,
                 "Depth corruption delta [mm]", cmap="coolwarm", vmin=-lim, vmax=lim)
    save_overlay(out / "depth_active_overlay.png", rgb, act,
                 "Active depth over RGB", cmap="viridis", vmin=.2, vmax=1.0)
    # Reference-error maps separate the model's native geometry error from
    # the synthetic corruption injected by DCR-E1-4.
    for ref_name, ref_depth in (("sensor", sensor), ("rendered", rendered)):
        if ref_depth is None:
            continue
        ref = np.squeeze(to_numpy(ref_depth))
        valid = np.isfinite(ref) & (ref > 0) & np.isfinite(nom) & (nom > 0)
        for src_name, src in (("pred_nominal", nom), ("active", act)):
            diff = np.full_like(ref, np.nan, dtype=np.float32)
            diff[valid] = (src[valid] - ref[valid]) * 1000.0
            finite = np.isfinite(diff)
            if finite.any():
                lim = max(1.0, float(np.nanpercentile(np.abs(diff[finite]), 99)))
                save_heatmap(
                    out / f"depth_{src_name}_minus_{ref_name}_mm.png",
                    diff, f"{src_name} - {ref_name} [mm]",
                    cmap="coolwarm", vmin=-lim, vmax=lim)

    for name, depth in (("pred_nominal", nom), ("active", act),
                        ("sensor", sensor), ("rendered", rendered)):
        if depth is None:
            continue
        d = np.squeeze(to_numpy(depth))
        if name in ("sensor", "rendered"):
            save_heatmap(out / f"depth_{name}.png", d, f"{name} depth", vmin=.2, vmax=1.0)
        pts, colors = depth_to_points(d, K, rgb, stride=point_stride)
        write_points_ply(out / f"pointcloud_{name}.ply", pts, colors)


def proposal_maps(proposal_logits: Any):
    x = torch.as_tensor(proposal_logits).detach().float()
    if x.ndim != 4 or x.shape[1] < 3:
        raise ValueError("Expected proposal logits [B,>=3,H,W]")
    obj = torch.softmax(x[:, :2], 1)[:, 1:2]
    grasp = torch.sigmoid(x[:, 2:3])
    return obj, grasp


def save_proposal_bundle(out_dir: str | Path, rgb: np.ndarray,
                         proposal_logits: Any) -> None:
    out = Path(out_dir)
    obj, grasp = proposal_maps(proposal_logits)
    save_heatmap(out / "objectness_fg.png", obj, "Objectness foreground", vmin=0, vmax=1)
    save_overlay(out / "objectness_overlay.png", rgb, obj, "Objectness", vmin=0, vmax=1)
    save_heatmap(out / "graspness.png", grasp, "DPT graspness", vmin=0, vmax=1)
    save_overlay(out / "graspness_overlay.png", rgb, grasp, "DPT graspness", vmin=0, vmax=1)


def save_spatial_bundle(out_dir: str | Path, rgb: np.ndarray,
                        spatial_aux: Mapping[str, Any]) -> None:
    out = Path(out_dir)
    keys = {
        "spatial_mean_z_map": ("spatial_mean_z.png", "GSE metric depth", "viridis", .2, 1.0),
        "spatial_gate_mean_map": ("spatial_gate.png", "GSE gate mean", "viridis", 0, 1),
        "spatial_delta_abs_map": ("spatial_delta_abs.png", "GSE |delta|", "magma", 0, None),
        "spatial_update_abs_map": ("spatial_update_abs.png", "GSE |gate*delta|", "magma", 0, None),
    }
    for key, (name, title, cmap, vmin, vmax) in keys.items():
        if key not in spatial_aux:
            continue
        save_heatmap(out / name, spatial_aux[key], title, cmap=cmap, vmin=vmin, vmax=vmax)
        save_overlay(out / name.replace(".png", "_overlay.png"), rgb,
                     spatial_aux[key], title, cmap=cmap, vmin=vmin, vmax=vmax)


def sparse_query_map(token_ids: Any, values: Any, hw: Tuple[int, int],
                     reduce: str = "max") -> np.ndarray:
    ids = np.asarray(to_numpy(token_ids), np.int64).reshape(-1)
    val = np.asarray(to_numpy(values), np.float32).reshape(-1)
    if len(ids) != len(val):
        raise ValueError("token/value length mismatch")
    h, w = map(int, hw)
    out = np.full(h * w, np.nan, np.float32)
    for idx, v in zip(ids, val):
        if not 0 <= int(idx) < h * w:
            continue
        old = out[int(idx)]
        if np.isnan(old):
            out[int(idx)] = v
        elif reduce == "min":
            out[int(idx)] = min(float(old), float(v))
        elif reduce == "mean":
            out[int(idx)] = .5 * (float(old) + float(v))
        else:
            out[int(idx)] = max(float(old), float(v))
    return out.reshape(h, w)


def save_query_scalar_overlay(path: str | Path, rgb: np.ndarray,
                              token_ids: Any, values: Any, title: str,
                              cmap: str = "coolwarm", symmetric: bool = False) -> None:
    h, w = rgb.shape[:2]
    arr = sparse_query_map(token_ids, values, (h, w), reduce="mean")
    masked = np.ma.masked_invalid(arr)
    kwargs = {}
    if symmetric:
        finite = np.isfinite(arr)
        lim = max(1e-6, float(np.nanpercentile(np.abs(arr[finite]), 99))) if finite.any() else 1.0
        kwargs.update(vmin=-lim, vmax=lim)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.imshow(rgb)
    im = ax.imshow(masked, cmap=cmap, alpha=.78, **kwargs)
    ax.axis("off")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout(pad=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_query_response(out_dir: str | Path, rgb: np.ndarray,
                        token_ids: Any, stage1_score: Any,
                        selected_offset_mm: Any,
                        local_utility: Any) -> None:
    out = Path(out_dir)
    h, w = rgb.shape[:2]
    score_map = sparse_query_map(token_ids, stage1_score, (h, w))
    off_map = sparse_query_map(token_ids, selected_offset_mm, (h, w), reduce="mean")
    util = np.asarray(to_numpy(local_utility))
    best = util.max(0) if util.ndim == 2 else util.reshape(-1)
    utility_map = sparse_query_map(token_ids, best, (h, w))
    for name, arr, title, cmap in (
        ("query_stage1_score.png", score_map, "Stage-1 score at query pixels", "viridis"),
        ("query_selected_offset_mm.png", off_map, "Selected center offset [mm]", "coolwarm"),
        ("query_best_local_utility.png", utility_map, "Best local CDF utility", "magma"),
    ):
        masked = np.ma.masked_invalid(arr)
        plt = _plt()
        fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
        ax.imshow(rgb)
        im = ax.imshow(masked, cmap=cmap, alpha=.75)
        ax.axis("off")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
        fig.tight_layout(pad=0)
        fig.savefig(out / name)
        plt.close(fig)


def _project_xyz(xyz: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = xyz[:, 2]
    valid = z > 1e-6
    uv = np.zeros((len(xyz), 2), np.float32)
    uv[:, 0] = K[0, 0] * xyz[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
    uv[:, 1] = K[1, 1] * xyz[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    return uv, valid


def save_grasp_scene_ply(path: str | Path, scene_points: Any,
                         scene_colors: Optional[Any], grasps: Any,
                         topk: int = 50, eval_scores: Optional[Any] = None,
                         collision: Optional[Any] = None,
                         gripper_points: int = 160) -> bool:
    """Write scene points plus sampled GraspGroup meshes to one colored PLY.

    Returns False when graspnetAPI/Open3D is unavailable; point-cloud-only PLY
    should be written separately by the caller in that case.
    """
    try:
        import open3d as o3d
        from graspnetAPI.grasp import GraspGroup
    except Exception:
        return False
    pts = np.asarray(scene_points, np.float32).reshape(-1, 3)
    if scene_colors is None:
        cols = np.full((len(pts), 3), .55, np.float32)
    else:
        cols = np.clip(np.asarray(scene_colors, np.float32).reshape(-1, 3), 0, 1)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    cloud.colors = o3d.utility.Vector3dVector(cols)

    g = np.asarray(to_numpy(grasps), np.float32).reshape(-1, 17)
    if len(g):
        order = np.argsort(-g[:, 0], kind="stable")[:min(int(topk), len(g))]
        selected = g[order]
        ev = None if eval_scores is None else np.asarray(eval_scores).reshape(-1)[order]
        co = None if collision is None else np.asarray(collision).reshape(-1)[order]
        gg = GraspGroup(selected.copy())
        for i, geom in enumerate(gg.to_open3d_geometry_list()):
            try:
                sample = geom.sample_points_uniformly(number_of_points=int(gripper_points))
            except Exception:
                continue
            if ev is None:
                t = i / max(1, len(selected) - 1)
                color = [float(t), float(.8 - .4 * t), float(1. - t)]
            else:
                score = float(ev[i])
                bad = bool(co[i]) if co is not None else False
                if bad or score <= 0:
                    color = [.30, .30, .30]
                elif score <= .4:
                    color = [.15, .80, .20]
                elif score <= .8:
                    color = [1.00, .75, .10]
                else:
                    color = [1.00, .30, .10]
            sample.paint_uniform_color(color)
            cloud += sample
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)
    return True


def save_grasp_overlay(path: str | Path, rgb: np.ndarray, grasps: Any, K: Any,
                       title: str = "", topk: int = 50,
                       eval_scores: Optional[Any] = None,
                       collision: Optional[Any] = None) -> None:
    import cv2
    image = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
    g = np.asarray(to_numpy(grasps), np.float32).reshape(-1, 17)
    if len(g) == 0:
        cv2.imwrite(str(path), image)
        return
    order = np.argsort(-g[:, 0], kind="stable")[:min(int(topk), len(g))]
    g = g[order]
    k = to_numpy(K)
    if k.ndim == 3:
        k = k[0]
    centers = g[:, 13:16]
    uv, visible = _project_xyz(centers, k)
    score_min, score_max = float(g[:, 0].min()), float(g[:, 0].max())
    den = max(score_max - score_min, 1e-6)
    eval_arr = None if eval_scores is None else np.asarray(eval_scores).reshape(-1)[order]
    coll_arr = None if collision is None else np.asarray(collision).reshape(-1)[order]
    for i, (row, p, ok) in enumerate(zip(g, uv, visible)):
        if not ok:
            continue
        u, v = int(round(p[0])), int(round(p[1]))
        if not (0 <= u < image.shape[1] and 0 <= v < image.shape[0]):
            continue
        if eval_arr is not None:
            ev = float(eval_arr[i])
            col = bool(coll_arr[i]) if coll_arr is not None else False
            if col or ev <= 0:
                color = (80, 80, 80)
            elif ev <= .4:
                color = (60, 210, 60)
            elif ev <= .8:
                color = (0, 210, 255)
            else:
                color = (0, 100, 255)
        else:
            t = (float(row[0]) - score_min) / den
            color = (int(255 * (1 - t)), int(100 + 155 * t), int(255 * t))
        radius = 2 if i >= 10 else 4
        cv2.circle(image, (u, v), radius, color, -1, cv2.LINE_AA)
        R = row[4:13].reshape(3, 3)
        approach = centers[i] + R[:, 0] * .035
        width_l = centers[i] - R[:, 1] * float(row[1]) * .5
        width_r = centers[i] + R[:, 1] * float(row[1]) * .5
        q3, qok = _project_xyz(np.stack((approach, width_l, width_r)), k)
        if qok.all():
            ap = tuple(np.round(q3[0]).astype(int))
            lp = tuple(np.round(q3[1]).astype(int))
            rp = tuple(np.round(q3[2]).astype(int))
            cv2.line(image, (u, v), ap, color, 1, cv2.LINE_AA)
            cv2.line(image, lp, rp, color, 1, cv2.LINE_AA)
    if title:
        cv2.putText(image, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 0), 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def save_local_patch_overlay(path: str | Path, rgb: np.ndarray,
                             debug: Mapping[str, Any],
                             max_queries: int = 8) -> None:
    import cv2
    img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
    puv = debug.get("kview_debug_patch_uv0")
    center = debug.get("kview_debug_center_uv0")
    attn = debug.get("kview_debug_patch_attn0")
    valid = debug.get("kview_debug_patch_valid0")
    if puv is None:
        cv2.imwrite(str(path), img)
        return
    p = to_numpy(puv)
    c = None if center is None else to_numpy(center)
    a = None if attn is None else to_numpy(attn)
    vm = None if valid is None else to_numpy(valid).astype(bool)
    nq = min(int(max_queries), p.shape[0])
    for qi in range(nq):
        pts = p[qi].reshape(-1, 2)
        weights = np.ones(len(pts), np.float32) if a is None else a[qi].reshape(-1)
        weights = (weights - weights.min()) / max(float(weights.max() - weights.min()), 1e-6)
        for j, ((u, v), ww) in enumerate(zip(pts, weights)):
            if vm is not None and not bool(vm[qi].reshape(-1)[j]):
                continue
            color = (int(255 * (1 - ww)), int(80 + 175 * ww), int(255 * ww))
            cv2.circle(img, (int(round(u)), int(round(v))), 1 + int(ww > .65),
                       color, -1, cv2.LINE_AA)
        if c is not None:
            u, v = np.round(c[qi]).astype(int)
            cv2.circle(img, (u, v), 4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"q{qi}", (u + 3, v - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        .35, (255, 255, 255), 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def save_center_cdf_panels(path: str | Path, cdf_logits: Any,
                           offsets_mm: Any, query_indices: Sequence[int],
                           selected: Optional[Any] = None) -> None:
    p = torch.sigmoid(torch.as_tensor(cdf_logits).detach().float()).cpu().numpy()
    off = np.asarray(to_numpy(offsets_mm)).reshape(-1)
    sel = None if selected is None else np.asarray(to_numpy(selected), np.int64).reshape(-1)
    qs = [int(q) for q in query_indices if 0 <= int(q) < p.shape[1]]
    if not qs:
        return
    plt = _plt()
    fig, axes = plt.subplots(len(qs), 2, figsize=(10, 3.4 * len(qs)),
                             dpi=160, squeeze=False)
    thresholds = [.2, .4, .6, .8, 1.0, 1.2]
    for row, q in enumerate(qs):
        im = axes[row, 0].imshow(p[:, q, :], aspect="auto", vmin=0, vmax=1,
                                 cmap="viridis")
        axes[row, 0].set_yticks(range(len(off)), [f"{x:g}" for x in off])
        axes[row, 0].set_xticks(range(6), [str(x) for x in thresholds])
        axes[row, 0].set_ylabel("center offset [mm]")
        axes[row, 0].set_title(f"q={q} CDF probabilities")
        fig.colorbar(im, ax=axes[row, 0], fraction=.035, pad=.02)
        utility = p[:, q].mean(-1)
        axes[row, 1].plot(off, utility, marker="o")
        if sel is not None:
            s = int(sel[q])
            axes[row, 1].scatter([off[s]], [utility[s]], s=55)
        axes[row, 1].set_ylim(0, 1)
        axes[row, 1].set_xlabel("center offset [mm]")
        axes[row, 1].set_ylabel("mean CDF utility")
        axes[row, 1].grid(True, alpha=.25)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_stage1_angle_depth(path: str | Path, stage1_ep: Mapping[str, Any],
                            query_ids: Any, query_indices: Sequence[int]) -> None:
    if "grasp_cdf_pred_angle_depth" not in stage1_ep:
        return
    raw = torch.as_tensor(stage1_ep["grasp_cdf_pred_angle_depth"]).detach().float()
    # [B,6,Q,A,D] in the current CDF implementation.
    utility = raw.sigmoid().mean(1)[0].cpu().numpy()
    qids = np.asarray(to_numpy(query_ids), np.int64).reshape(-1)
    qs = [int(q) for q in query_indices if 0 <= int(q) < len(qids)]
    if not qs:
        return
    plt = _plt()
    fig, axes = plt.subplots(len(qs), 1, figsize=(6, 3.6 * len(qs)),
                             dpi=160, squeeze=False)
    for row, q in enumerate(qs):
        qid = int(qids[q])
        if not 0 <= qid < utility.shape[0]:
            continue
        im = axes[row, 0].imshow(utility[qid], aspect="auto", vmin=0, vmax=1,
                                 cmap="viridis")
        axes[row, 0].set_xlabel("insertion-depth index")
        axes[row, 0].set_ylabel("in-plane angle index")
        axes[row, 0].set_title(f"Stage-1 angle-depth utility: query {q} / id {qid}")
        fig.colorbar(im, ax=axes[row, 0], fraction=.035, pad=.02)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_view_response(out_dir: str | Path, rgb: np.ndarray,
                       stage1_ep: Mapping[str, Any],
                       bundle: Mapping[str, Any]) -> None:
    if "view_score" not in stage1_ep:
        return
    score = torch.as_tensor(stage1_ep["view_score"]).detach().float()
    if score.ndim != 3:
        return
    prob = torch.softmax(score, -1)
    entropy = -(prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()).sum(-1)
    entropy = entropy / math.log(max(score.shape[-1], 2))
    top2 = torch.topk(prob, k=min(2, prob.shape[-1]), dim=-1).values
    margin = top2[..., 0] - top2[..., -1]
    # view_score is normally defined on the base seed set, whereas
    # token_sel_idx may already be expanded into CVA queries.  Prefer an
    # explicitly aligned source before falling back to the expanded tensor.
    bundle_token = torch.as_tensor(bundle["token_ids"])
    if entropy.shape[1] == int(bundle_token.numel()):
        token = bundle_token.reshape(-1)
    else:
        source = stage1_ep.get(
            "kview_base_token_sel_idx",
            stage1_ep.get("token_sel_idx", bundle["token_ids"]))
        token = torch.as_tensor(source)
        token = token[0] if token.ndim == 2 else token
        token = token.reshape(-1)
    n = min(len(token), entropy.shape[1])
    h, w = rgb.shape[:2]
    for name, val, title in (
        ("view_entropy.png", entropy[0, :n], "View distribution entropy"),
        ("view_margin.png", margin[0, :n], "View top1-top2 probability margin"),
    ):
        arr = sparse_query_map(token[:n], val, (h, w))
        masked = np.ma.masked_invalid(arr)
        plt = _plt()
        fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
        ax.imshow(rgb)
        im = ax.imshow(masked, cmap="magma", alpha=.75)
        ax.axis("off")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
        fig.tight_layout(pad=0)
        fig.savefig(Path(out_dir) / name)
        plt.close(fig)


def save_center_selection_motion(path: str | Path, rgb: np.ndarray, K: Any,
                                 actions: Any, selected: Any, zero: int,
                                 offsets_mm: Any, max_lines: int = 160,
                                 title: str = "Native to selected center") -> Dict[str, float]:
    """Visualize the physical translation made by a center corrector."""
    import cv2
    a = np.asarray(to_numpy(actions), np.float32)
    sel = np.asarray(to_numpy(selected), np.int64).reshape(-1)
    off = np.asarray(to_numpy(offsets_mm), np.float32).reshape(-1)
    if a.ndim != 3 or a.shape[-1] != 17 or a.shape[1] != len(sel):
        raise ValueError("Expected center actions [C,Q,17] and selected [Q]")
    q = np.arange(len(sel))
    native = a[int(zero), q, 13:16]
    chosen = a[sel, q, 13:16]
    chosen_off = off[sel]
    moved = np.flatnonzero(sel != int(zero))
    image = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
    k = to_numpy(K)
    if k.ndim == 3:
        k = k[0]
    if len(moved):
        mag = np.linalg.norm(chosen[moved] - native[moved], axis=1)
        order = moved[np.argsort(-mag)[:min(int(max_lines), len(moved))]]
        for qi in order:
            uv, ok = _project_xyz(np.stack((native[qi], chosen[qi])), k)
            if not ok.all():
                continue
            p0 = tuple(np.round(uv[0]).astype(int))
            p1 = tuple(np.round(uv[1]).astype(int))
            # blue = toward camera / negative z offset, red = away / positive.
            color = (230, 100, 30) if chosen_off[qi] < 0 else (30, 80, 230)
            cv2.line(image, p0, p1, color, 1, cv2.LINE_AA)
            cv2.circle(image, p0, 2, (235, 235, 235), -1, cv2.LINE_AA)
            cv2.circle(image, p1, 3, color, -1, cv2.LINE_AA)
    text = (
        f"{title} | move={len(moved)/max(1,len(sel)):.2f} "
        f"mean dz={float(chosen_off.mean()):+.1f} mm")
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                .45, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                .45, (0,0,0), 1, cv2.LINE_AA)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return {
        "queries": int(len(sel)),
        "move_rate": float(len(moved) / max(1, len(sel))),
        "negative_rate": float((chosen_off < 0).mean()),
        "positive_rate": float((chosen_off > 0).mean()),
        "mean_offset_mm": float(chosen_off.mean()),
        "mean_abs_offset_mm": float(np.abs(chosen_off).mean()),
    }


def save_corruption_motion(path: str | Path, rgb: np.ndarray, K: Any,
                           nominal_bundle: Mapping[str, Any],
                           active_bundle: Mapping[str, Any],
                           max_lines: int = 128) -> Dict[str, float]:
    """Match Stage-1 queries by image token and visualize native-center motion."""
    nom_tok = np.asarray(to_numpy(nominal_bundle["token_ids"]), np.int64).reshape(-1)
    act_tok = np.asarray(to_numpy(active_bundle["token_ids"]), np.int64).reshape(-1)
    nom = np.asarray(to_numpy(nominal_bundle["native"]), np.float32).reshape(-1, 17)
    act = np.asarray(to_numpy(active_bundle["native"]), np.float32).reshape(-1, 17)
    amap = {int(t): i for i, t in enumerate(act_tok)}
    pairs = [(i, amap[int(t)]) for i, t in enumerate(nom_tok) if int(t) in amap]
    if not pairs:
        return {"matched_queries": 0, "mean_center_shift_mm": float("nan")}
    shift = np.asarray([
        np.linalg.norm(act[j, 13:16] - nom[i, 13:16]) * 1000.
        for i, j in pairs
    ])
    import cv2
    img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
    k = to_numpy(K)
    if k.ndim == 3:
        k = k[0]
    ids = np.argsort(-shift)[:min(int(max_lines), len(pairs))]
    for rank in ids:
        i, j = pairs[int(rank)]
        p, ok = _project_xyz(np.stack((nom[i, 13:16], act[j, 13:16])), k)
        if not ok.all():
            continue
        a = tuple(np.round(p[0]).astype(int))
        b = tuple(np.round(p[1]).astype(int))
        cv2.line(img, a, b, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.circle(img, a, 2, (255, 120, 0), -1)
        cv2.circle(img, b, 2, (0, 80, 255), -1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return {
        "matched_queries": int(len(pairs)),
        "match_ratio_nominal": float(len(pairs) / max(1, len(nom_tok))),
        "mean_center_shift_mm": float(shift.mean()),
        "median_center_shift_mm": float(np.median(shift)),
        "p90_center_shift_mm": float(np.percentile(shift, 90)),
    }


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True),
                    encoding="utf-8")


def make_contact_sheet(path: str | Path, images: Sequence[Tuple[str, Path]],
                       cols: int = 3, thumb_width: int = 420) -> None:
    from PIL import Image, ImageDraw
    loaded = []
    for title, p in images:
        if not Path(p).is_file():
            continue
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        scale = thumb_width / max(1, im.width)
        im = im.resize((thumb_width, max(1, int(im.height * scale))))
        loaded.append((title, im))
    if not loaded:
        return
    rows = int(math.ceil(len(loaded) / max(1, cols)))
    cell_h = max(im.height for _, im in loaded) + 32
    canvas = Image.new("RGB", (cols * thumb_width, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, im) in enumerate(loaded):
        x = (i % cols) * thumb_width
        y = (i // cols) * cell_h
        canvas.paste(im, (x, y + 28))
        draw.text((x + 4, y + 4), title, fill="black")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_html_index(path: str | Path, title: str,
                     groups: Mapping[str, Sequence[Tuple[str, Path]]],
                     base_dir: Optional[Path] = None) -> None:
    path = Path(path)
    base = base_dir or path.parent
    blocks = []
    for group, items in groups.items():
        cards = []
        for label, p in items:
            p = Path(p)
            if not p.is_file():
                continue
            try:
                rel = p.relative_to(base)
            except ValueError:
                rel = p
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                cards.append(
                    f'<figure><img loading="lazy" src="{html.escape(str(rel))}">'
                    f'<figcaption>{html.escape(label)}</figcaption></figure>')
            else:
                cards.append(
                    f'<p><a href="{html.escape(str(rel))}">{html.escape(label)}</a></p>')
        if cards:
            blocks.append(f"<section><h2>{html.escape(group)}</h2>"
                          f'<div class="grid">{"".join(cards)}</div></section>')
    doc = f"""<!doctype html><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;max-width:1500px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}}
figure{{margin:0;border:1px solid #ccc;padding:8px}} img{{width:100%;height:auto}}
figcaption{{font-size:13px;margin-top:4px}} h2{{margin-top:28px}}
</style><h1>{html.escape(title)}</h1>{"".join(blocks)}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
