#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import scipy.io as scio
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.data_utils import (
    get_workspace_mask,
    CameraInfo,
    create_point_cloud_from_depth_image,
)


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_gray(path):
    return np.array(Image.open(path))


def load_graspness(path):
    g = np.load(path)
    g = np.asarray(g, dtype=np.float32)

    if g.ndim == 2 and g.shape[1] == 1:
        g = g[:, 0]
    elif g.ndim == 2 and g.shape[0] == 1:
        g = g[0]
    elif g.ndim != 1:
        g = g.reshape(g.shape[0], -1)[:, 0]

    return g.reshape(-1).astype(np.float32)


def safe_float(x):
    return float(np.asarray(x).reshape(-1)[0])


def safe_minmax(x, valid=None, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(x)
    else:
        valid = valid & np.isfinite(x)

    out = np.zeros_like(x, dtype=np.float32)
    if valid.sum() == 0:
        return out

    v = x[valid]
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin < eps:
        if vmax > eps:
            out[valid] = 1.0
        else:
            out[valid] = 0.0
        return out

    out[valid] = (v - vmin) / (vmax - vmin)
    return out


def get_paths(args, scene_name, ann_id):
    scene_dir = Path(args.dataset_root) / "scenes" / scene_name / args.camera_type

    rgb_path = scene_dir / "rgb" / f"{ann_id:04d}.png"
    label_path = scene_dir / "label" / f"{ann_id:04d}.png"
    meta_path = scene_dir / "meta" / f"{ann_id:04d}.mat"

    if args.depth_type == "real":
        depth_path = scene_dir / "depth" / f"{ann_id:04d}.png"
        scene_g_root = Path(args.scene_graspness_dir) if args.scene_graspness_dir else Path(args.dataset_root) / "graspness"
        inst_g_root = Path(args.instance_graspness_dir) if args.instance_graspness_dir else Path(args.dataset_root) / "graspness_instance"
    elif args.depth_type == "virtual":
        depth_path = Path(args.virtual_dataset_root) / scene_name / args.camera_type / f"{ann_id:04d}_depth.png"
        scene_g_root = Path(args.scene_graspness_dir) if args.scene_graspness_dir else Path(args.dataset_root) / "virtual_graspness"
        inst_g_root = Path(args.instance_graspness_dir) if args.instance_graspness_dir else Path(args.dataset_root) / "virtual_graspness_instance"
    else:
        raise ValueError(f"Unsupported depth_type: {args.depth_type}")

    scene_g_path = scene_g_root / scene_name / args.camera_type / f"{ann_id:04d}.npy"
    inst_g_path = inst_g_root / scene_name / args.camera_type / f"{ann_id:04d}.npy"

    return {
        "scene_dir": scene_dir,
        "rgb": rgb_path,
        "depth": depth_path,
        "label": label_path,
        "meta": meta_path,
        "scene_g": scene_g_path,
        "inst_g": inst_g_path,
    }


def build_masks(args, scene_dir, depth, seg, meta, ann_id):
    intrinsic = meta["intrinsic_matrix"]
    factor_depth = safe_float(meta["factor_depth"])

    camera = CameraInfo(
        1280.0,
        720.0,
        intrinsic[0][0],
        intrinsic[1][1],
        intrinsic[0][2],
        intrinsic[1][2],
        factor_depth,
    )

    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    depth_mask = depth > 0

    camera_poses = np.load(scene_dir / "camera_poses.npy")
    camera_pose = camera_poses[ann_id]
    align_mat = np.load(scene_dir / "cam0_wrt_table.npy")
    trans = np.dot(align_mat, camera_pose)

    workspace_mask = get_workspace_mask(
        cloud, seg, trans=trans, organized=True, outlier=args.outlier
    )
    workspace_mask = depth_mask & workspace_mask

    return {
        "workspace": workspace_mask,
        "depth": depth_mask,
    }


def choose_mask(masks, target_len, preferred="workspace"):
    if preferred in masks and int(masks[preferred].sum()) == target_len:
        return preferred, masks[preferred]

    for name, m in masks.items():
        if int(m.sum()) == target_len:
            return name, m

    lens = {k: int(v.sum()) for k, v in masks.items()}
    raise ValueError(
        f"Cannot find matched mask for graspness length={target_len}. "
        f"candidate mask lengths={lens}"
    )


def scatter_graspness_to_map(graspness, mask, image_shape):
    H, W = image_shape[:2]
    if graspness.shape[0] != int(mask.sum()):
        raise ValueError(
            f"graspness length mismatch: graspness={graspness.shape[0]}, "
            f"mask_sum={int(mask.sum())}"
        )

    gmap = np.full((H, W), np.nan, dtype=np.float32)
    gmap[mask] = graspness
    return gmap


def get_colormap(name):
    name = name.lower()
    if name == "jet":
        return cv2.COLORMAP_JET
    if name == "viridis":
        return cv2.COLORMAP_VIRIDIS
    if name == "plasma":
        return cv2.COLORMAP_PLASMA
    if name == "turbo":
        return getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)


def make_heatmap_panel(rgb, gmap, valid, args):
    if args.vis_norm == "fixed":
        norm = np.zeros_like(gmap, dtype=np.float32)
        norm[valid] = (gmap[valid] - args.vmin) / (args.vmax - args.vmin + 1e-8)
        norm = np.clip(norm, 0.0, 1.0)
    elif args.vis_norm == "per_image":
        norm = safe_minmax(gmap, valid)
    else:
        raise ValueError(f"Unsupported vis_norm: {args.vis_norm}")

    heat_u8 = (norm * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(heat_u8, get_colormap(args.colormap))
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    # graspness panel: dim RGB background, heatmap only on valid points
    bg = (rgb.astype(np.float32) * args.background_dim).astype(np.uint8)
    panel = bg.copy()
    panel[valid] = heat[valid]
    return panel, heat


def make_overlap_panel(rgb, heat, valid, alpha=0.55):
    out = rgb.copy().astype(np.float32)
    out[valid] = (1.0 - alpha) * out[valid] + alpha * heat[valid].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_title(img, title, subtitle=None):
    out = img.copy()
    H, W = out.shape[:2]
    bar_h = 54 if subtitle else 34

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (W, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

    cv2.putText(
        out, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2, cv2.LINE_AA
    )
    if subtitle:
        cv2.putText(
            out, subtitle, (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (255, 255, 255), 1, cv2.LINE_AA
        )
    return out


def resize_panel(img, width):
    H, W = img.shape[:2]
    scale = width / float(W)
    new_h = max(1, int(round(H * scale)))
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)


def stats_str(gmap, valid):
    if valid.sum() == 0:
        return "valid=0"
    v = gmap[valid]
    return (
        f"valid={int(valid.sum())} "
        f"min={np.nanmin(v):.3f} "
        f"mean={np.nanmean(v):.3f} "
        f"p95={np.nanpercentile(v, 95):.3f} "
        f"max={np.nanmax(v):.3f}"
    )


def stats_row(scene_name, ann_id, mode, gmap, valid):
    if valid.sum() == 0:
        return {
            "scene": scene_name,
            "ann_id": ann_id,
            "mode": mode,
            "valid": 0,
            "min": np.nan,
            "mean": np.nan,
            "p50": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "max": np.nan,
            "nonzero_ratio": np.nan,
        }

    v = gmap[valid]
    return {
        "scene": scene_name,
        "ann_id": ann_id,
        "mode": mode,
        "valid": int(valid.sum()),
        "min": float(np.nanmin(v)),
        "mean": float(np.nanmean(v)),
        "p50": float(np.nanpercentile(v, 50)),
        "p90": float(np.nanpercentile(v, 90)),
        "p95": float(np.nanpercentile(v, 95)),
        "max": float(np.nanmax(v)),
        "nonzero_ratio": float(np.mean(v > 1e-6)),
    }


def make_compare_image(rgb, scene_map, inst_map, mask, args):
    valid_scene = mask & np.isfinite(scene_map)
    valid_inst = mask & np.isfinite(inst_map)

    scene_panel, scene_heat = make_heatmap_panel(rgb, scene_map, valid_scene, args)
    scene_overlap = make_overlap_panel(rgb, scene_heat, valid_scene, alpha=args.alpha)

    inst_panel, inst_heat = make_heatmap_panel(rgb, inst_map, valid_inst, args)
    inst_overlap = make_overlap_panel(rgb, inst_heat, valid_inst, alpha=args.alpha)

    rgb_scene = draw_title(rgb, "RGB", "scene-level row")
    scene_panel = draw_title(scene_panel, "Scene-level graspness", stats_str(scene_map, valid_scene))
    scene_overlap = draw_title(scene_overlap, "Scene-level overlap", f"alpha={args.alpha:.2f}")

    rgb_inst = draw_title(rgb, "RGB", "instance-level row")
    inst_panel = draw_title(inst_panel, "Instance-level graspness", stats_str(inst_map, valid_inst))
    inst_overlap = draw_title(inst_overlap, "Instance-level overlap", f"alpha={args.alpha:.2f}")

    panels = [
        resize_panel(rgb_scene, args.panel_width),
        resize_panel(scene_panel, args.panel_width),
        resize_panel(scene_overlap, args.panel_width),
        resize_panel(rgb_inst, args.panel_width),
        resize_panel(inst_panel, args.panel_width),
        resize_panel(inst_overlap, args.panel_width),
    ]

    row1 = np.concatenate(panels[:3], axis=1)
    row2 = np.concatenate(panels[3:], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    return grid


def write_csv(csv_path, rows):
    if len(rows) == 0:
        return
    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="/data/robotarm/dataset/graspnet")
    parser.add_argument("--virtual_dataset_root", default=None)
    parser.add_argument("--camera_type", default="realsense", choices=["realsense", "kinect"])
    parser.add_argument("--depth_type", default="real", choices=["real", "virtual"])

    parser.add_argument("--scene_start", type=int, default=0)
    parser.add_argument("--scene_end", type=int, default=100,
                        help="Inclusive scene id end. Use 99 if you only want train scenes.")
    parser.add_argument("--anno_ids", type=int, nargs="+", default=[0, 128, 255])

    parser.add_argument("--scene_graspness_dir", default=None,
                        help="Override scene-level graspness root. Default: dataset_root/graspness")
    parser.add_argument("--instance_graspness_dir", default=None,
                        help="Override instance-level graspness root. Default: dataset_root/graspness_instance")

    parser.add_argument("--mask_mode", default="workspace", choices=["workspace", "depth"],
                        help="Preferred mask mode. Auto fallback is used if length mismatch.")
    parser.add_argument("--outlier", type=float, default=0.02)

    parser.add_argument("--save_root", default=None)
    parser.add_argument("--panel_width", type=int, default=640)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--background_dim", type=float, default=0.25)

    parser.add_argument("--vis_norm", default="fixed", choices=["fixed", "per_image"],
                        help="fixed preserves [0,1] comparability; per_image stretches each map.")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--colormap", default="turbo", choices=["turbo", "jet", "viridis", "plasma"])

    parser.add_argument("--strict", action="store_true",
                        help="Raise error on missing/mismatch instead of skipping.")
    args = parser.parse_args()

    if args.virtual_dataset_root is None:
        args.virtual_dataset_root = os.path.join(args.dataset_root, "virtual_scenes")

    if args.save_root is None:
        args.save_root = os.path.join(args.dataset_root, "graspness_compare_vis")

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    rows = []
    scene_ids = list(range(args.scene_start, args.scene_end + 1))

    for scene_id in tqdm(scene_ids, desc="Scenes"):
        scene_name = f"scene_{scene_id:04d}"

        for ann_id in args.anno_ids:
            try:
                paths = get_paths(args, scene_name, ann_id)

                required = ["rgb", "depth", "label", "meta", "scene_g", "inst_g"]
                missing = [k for k in required if not paths[k].exists()]
                if missing:
                    msg = f"[SKIP] {scene_name} ann={ann_id:04d}, missing: {[(k, str(paths[k])) for k in missing]}"
                    if args.strict:
                        raise FileNotFoundError(msg)
                    print(msg)
                    continue

                rgb = load_rgb(paths["rgb"])
                depth = load_gray(paths["depth"])
                seg = load_gray(paths["label"])
                meta = scio.loadmat(paths["meta"])

                scene_g = load_graspness(paths["scene_g"])
                inst_g = load_graspness(paths["inst_g"])

                if scene_g.shape[0] != inst_g.shape[0]:
                    msg = (
                        f"[SKIP] {scene_name} ann={ann_id:04d}: "
                        f"scene_g len={scene_g.shape[0]}, inst_g len={inst_g.shape[0]}"
                    )
                    if args.strict:
                        raise ValueError(msg)
                    print(msg)
                    continue

                masks = build_masks(args, paths["scene_dir"], depth, seg, meta, ann_id)
                mask_name, mask = choose_mask(masks, target_len=scene_g.shape[0], preferred=args.mask_mode)

                scene_map = scatter_graspness_to_map(scene_g, mask, rgb.shape)
                inst_map = scatter_graspness_to_map(inst_g, mask, rgb.shape)

                grid = make_compare_image(rgb, scene_map, inst_map, mask, args)

                out_dir = save_root / args.camera_type / scene_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{ann_id:04d}_scene_vs_instance_graspness.png"

                cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

                valid = mask & np.isfinite(scene_map) & np.isfinite(inst_map)
                diff_abs = float(np.nanmean(np.abs(inst_map[valid] - scene_map[valid]))) if valid.sum() > 0 else np.nan

                row_scene = stats_row(scene_name, ann_id, "scene", scene_map, mask & np.isfinite(scene_map))
                row_scene["mask_mode_used"] = mask_name
                row_scene["image_path"] = str(out_path)
                row_scene["abs_diff_to_other"] = diff_abs
                rows.append(row_scene)

                row_inst = stats_row(scene_name, ann_id, "instance", inst_map, mask & np.isfinite(inst_map))
                row_inst["mask_mode_used"] = mask_name
                row_inst["image_path"] = str(out_path)
                row_inst["abs_diff_to_other"] = diff_abs
                rows.append(row_inst)

            except Exception as e:
                msg = f"[ERROR] {scene_name} ann={ann_id:04d}: {repr(e)}"
                if args.strict:
                    raise
                print(msg)
                continue

    csv_path = save_root / "graspness_vis_stats.csv"
    write_csv(csv_path, rows)
    print(f"Done. Images saved to: {save_root}")
    print(f"Stats saved to: {csv_path}")


if __name__ == "__main__":
    main()