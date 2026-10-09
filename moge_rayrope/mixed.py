"""Source-explicit paired GraspNet RealSense / GN-Trans RGB training.

This module changes the **depth supervision target**, never the RGB-only model
input. RealSense: original RGB + full TSDF depth. GN-Trans: rendered RGB +
full rendered (virtual_scenes) depth. No fallback to fused-background targets.

The two domains are sampled at identical scene/frame indices. The old
GraspNetTransDataset implementation uses virtual depth for its preprocessing
point cloud, but this point cloud must not be passed as a network input.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import ConcatDataset, Subset

from dataset.graspnet_dataset import GraspNetMultiDataset, GraspNetTransDataset
from dataset.cdf_label_adapter import CVAExtendedLabelAdapter


class RealSenseFullTSDF(GraspNetMultiDataset):
    """Overrides only the GT target composer; no RGB/model changes."""

    def build_fused_gt_depth_m(self, gt_depth_raw, seg_raw, index, factor_depth):
        if not self.use_fuse_depth:
            raise RuntimeError("Full TSDF requires the explicit TSDF path")
        path = Path(self.fusedepthpath[index])
        if not path.is_file():
            raise FileNotFoundError(f"Required RealSense full TSDF depth: {path}")
        tsdf = np.asarray(Image.open(path))
        if tsdf.ndim != 2 or tsdf.shape != gt_depth_raw.shape or tsdf.shape != seg_raw.shape:
            raise ValueError(f"TSDF/raw/mask depth size mismatch: {path}")
        fd = float(self.gt_factor_depth) if self.gt_factor_depth is not None else float(factor_depth)
        if not np.isfinite(fd) or fd <= 0:
            raise ValueError(f"Invalid depth scale for {path}: {fd}")
        return tsdf.astype(np.float32) / fd


class GNTransFullRendered(GraspNetTransDataset):
    """Rendered RGB and full virtual depth (objects AND background)."""

    def build_fused_gt_depth_m(self, gt_depth_raw, seg_raw, index, factor_depth):
        if self.use_fuse_depth:
            raise RuntimeError("GN-Trans full-rendered policy forbids TSDF background substitution")
        if gt_depth_raw.ndim != 2 or gt_depth_raw.shape != seg_raw.shape:
            raise ValueError(f"GN-Trans rendered depth/mask size mismatch at {index}")
        fd = float(self.gt_factor_depth) if self.gt_factor_depth is not None else float(factor_depth)
        if not np.isfinite(fd) or fd <= 0:
            raise ValueError(f"Invalid GN-Trans depth scale for {index}: {fd}")
        return gt_depth_raw.astype(np.float32) / fd


def build_bases(root, gntrans_rgb_root, split, cfg, labels=True):
    common = dict(camera="realsense", split=split, num_points=20000,
                  voxel_size=.005, remove_outlier=True, augment=False,
                  load_label=bool(labels), min_depth=cfg.min_depth,
                  max_depth=cfg.max_depth, bin_num=256, depth_strides=1)
    real = RealSenseFullTSDF(
        root, use_gt_depth=False, use_fuse_depth=True,
        graspness_mode="scene", extend_angle=True,
        load_grasp_payload=False, **common)
    trans = GNTransFullRendered(
        root, gntrans_rgb_root, use_gt_depth=True, **common)
    # GN-Trans predates the CDF label adapter.
    trans.extend_angle = True
    trans.load_grasp_payload = False
    trans.use_fuse_depth = False
    return real, trans


def paired_frame_indices(real, trans, fraction, max_pairs=0):
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    step = round(1.0 / fraction)
    if abs(fraction * step - 1.0) > 1e-7:
        raise ValueError("Use inverse-integer frame fractions for deterministic schedules")
    def rows(dataset):
        return {(str(s), int(f)): i for i, (s, f) in
                enumerate(zip(dataset.scenename, dataset.frameid))
                if int(f) % step == 0}
    ra, tb = rows(real), rows(trans)
    if len(ra) != len(tb) or set(ra) != set(tb):
        raise RuntimeError("GraspNet/GN-Trans paired scene/frame schedule mismatch")
    keys = sorted(ra, key=lambda x: (x[0], x[1]))
    if max_pairs:
        keys = keys[:max_pairs]
    return [ra[k] for k in keys], [tb[k] for k in keys], [
        [int(k[0].rsplit("_", 1)[-1]), k[1]] for k in keys]


def check_depth_sources(real, trans, real_idx, trans_idx):
    """Fail early on absent inputs; never silently substitute another GT."""
    for ds, idx, color, gt in (
        (real, real_idx, real.colorpath, real.fusedepthpath),
        (trans, trans_idx, trans.colorpath, trans.gtdepthpath),
    ):
        for i in idx:
            for path in (color[i], gt[i], ds.labelpath[i], ds.metapath[i]):
                if not Path(path).is_file():
                    raise FileNotFoundError(f"Paired-depth data contract requires {path}")


def make_mixed_dataset(root, gntrans_rgb_root, split, fraction, labels, cfg,
                       label_folder, max_frames=0, include_trans=True):
    """Return (raw_bases, dataset, index_schedule).

    max_frames truncates the *combined* dataset in balanced paired units.
    Validation uses include_trans=False and thus RealSense/TSDF only.
    """
    real, trans = build_bases(root, gntrans_rgb_root, split, cfg, labels)
    if max_frames and include_trans and max_frames % 2:
        raise ValueError("Mixed max_frames must be even (equal source counts)")
    max_pairs = (max_frames // 2 if include_trans else max_frames) if max_frames else 0
    ri, ti, schedule = paired_frame_indices(real, trans, fraction, max_pairs)
    check_depth_sources(real, trans, ri, ti if include_trans else [])
    if labels:
        def adapter(ds):
            return CVAExtendedLabelAdapter(
                ds, dataset_root=root, use_cdf=True, label_folder=label_folder,
                num_angle=12, num_depth=4)
        real_data, trans_data = adapter(real), adapter(trans)
    else:
        real_data, trans_data = real, trans
    rs = Subset(real_data, ri)
    if not include_trans:
        return (real, trans), rs, [["realsense", *p] for p in schedule]
    gt = Subset(trans_data, ti)
    return ((real, trans), ConcatDataset([rs, gt]),
            [["realsense", *p] for p in schedule] +
            [["gntrans", *p] for p in schedule])


DEPTH_CONTRACT = {
    "realsense_rgb": "GraspNet/scenes/<scene>/realsense/rgb",
    "realsense_supervision": "FULL GraspNet tsdf_depth/<scene>/realsense/<frame>_depth.png",
    "gntrans_rgb": "GN-Trans/scenes/<five-digit-scene>/<frame>_color.png",
    "gntrans_supervision": "FULL GraspNet virtual_scenes/<scene>/realsense/<frame>_depth.png",
    "observed_depth_network_input": False,
    "background_fusion": False,
    "paired_frame_indices": True,
}
