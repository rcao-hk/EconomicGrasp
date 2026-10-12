#!/usr/bin/env python3
"""Audit paired RealSense (rendered foreground + TSDF background) and
GN-Trans (fully rendered) depth supervision.

Compare ACTUAL composed training targets in native coordinates and separately
report the raw TSDF-versus-rendered discrepancy. Verify that sampled dataset
gt_depth_m maps exactly to each source composition through its own crop/K.
No model weights, evaluator or GPU are required.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from scipy.io import loadmat


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--gntrans-rgb-root", required=True)
    p.add_argument("--split", choices=["train", "test_seen", "test_similar", "test_novel"],
                   default="train")
    p.add_argument("--fraction", type=float, default=.1)
    p.add_argument("--max-pairs", type=int, default=0, help="0 = all paired 10%% frames")
    p.add_argument("--loader-audit-pairs", type=int, default=8,
                   help="Actual crop and K check using dataset.get_data_label; 0 skips")
    p.add_argument("--output-prefix", required=True)
    return p


def _load_raw_depth(path):
    x = np.asarray(Image.open(path))
    if x.ndim != 2 or not np.issubdtype(x.dtype, np.number):
        raise ValueError(f"Non-2D or unsupported depth {path}: {x.shape} / {x.dtype}")
    return x


def _cropped_target(target_m, crop_box, hw):
    x0, y0, x1, y1 = [int(v) for v in crop_box]
    h, w = hw
    if x0 < 0 or y0 < 0 or x1 > target_m.shape[1] or y1 > target_m.shape[0]:
        raise ValueError(f"Crop outside depth image: {(x0,y0,x1,y1)}")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Empty crop")
    return cv2.resize(target_m[y0:y1,x0:x1], (w,h),
                      interpolation=cv2.INTER_NEAREST).astype(np.float32)


def _load_rgb_size(path):
    with Image.open(path) as im:
        return im.size[::-1]


def _stats(depth, mask):
    x = depth[mask & np.isfinite(depth) & (depth > 0)]
    return dict(n=int(x.size), mean_m=float(x.mean()) if x.size else None,
                p95_m=float(np.quantile(x, .95)) if x.size else None)


def _compare(a, b, mask):
    va = mask & np.isfinite(a) & (a > 0)
    vb = mask & np.isfinite(b) & (b > 0)
    both = va & vb
    error = np.abs(a[both] - b[both])
    denom = int(np.count_nonzero(va | vb))
    return dict(valid_a=float(va.mean()), valid_b=float(vb.mean()),
                valid_iou=(float(np.count_nonzero(both) / denom) if denom else None),
                overlap_count=int(error.size),
                abs_diff_mean_m=(float(error.mean()) if error.size else None),
                abs_diff_p90_m=(float(np.quantile(error, .9)) if error.size else None),
                abs_diff_p99_m=(float(np.quantile(error, .99)) if error.size else None))


def main():
    a = args_parser().parse_args()
    sys.argv = [sys.argv[0]]
    from moge_rayrope.config import ModelConfig
    from moge_rayrope.runtime import configure_main
    from moge_rayrope.mixed import build_bases, paired_frame_indices, check_depth_sources, DEPTH_CONTRACT

    if min(a.max_pairs, a.loader_audit_pairs) < 0:
        raise ValueError("Invalid frame limit")
    mc = ModelConfig()
    configure_main(mc, a.dataset_root,
                   "economic_grasp_label_300views_extend_angle_cdf_depth",
                   use_fuse_depth=True)
    real, trans = build_bases(a.dataset_root, a.gntrans_rgb_root, a.split, mc,
                              labels=(a.loader_audit_pairs > 0))
    real_idx, trans_idx, schedule = paired_frame_indices(
        real, trans, a.fraction, a.max_pairs)
    check_depth_sources(real, trans, real_idx, trans_idx)
    rows = []
    crop_mismatches = 0
    k_mismatches = 0
    for n, (ir, it, (sid, frame)) in enumerate(zip(real_idx, trans_idx, schedule)):
        meta = loadmat(real.metapath[ir])
        fd = float(np.asarray(meta["factor_depth"]).reshape(-1)[0])
        if not np.isfinite(fd) or fd <= 0:
            raise ValueError(f"Bad factor_depth: scene {sid}, frame {frame}")
        rs_tsdf_raw = _load_raw_depth(real.fusedepthpath[ir])
        rs_render_raw = _load_raw_depth(real.gtdepthpath[ir])
        gn_render_raw = _load_raw_depth(trans.gtdepthpath[it])
        seg = np.asarray(Image.open(real.labelpath[ir]))
        if (rs_tsdf_raw.shape != rs_render_raw.shape or
                rs_render_raw.shape != gn_render_raw.shape or
                gn_render_raw.shape != seg.shape):
            raise ValueError(
                f"Unaligned depth/segmentation scene {sid}:{frame}: "
                f"{rs_tsdf_raw.shape}/{rs_render_raw.shape}/"
                f"{gn_render_raw.shape}/{seg.shape}")
        rgb_rs = _load_rgb_size(real.colorpath[ir])
        rgb_gn = _load_rgb_size(trans.colorpath[it])
        if rgb_rs != seg.shape or rgb_gn != seg.shape:
            raise ValueError(
                f"RGB/depth dimensions differ at {sid}:{frame}: "
                f"{rgb_rs}/{rgb_gn}/{seg.shape}")

        # The *actual* training supervision: per-source compose policies.
        rs = real.build_fused_gt_depth_m(rs_render_raw, seg, ir, fd)
        gn = trans.build_fused_gt_depth_m(gn_render_raw, seg, it, fd)
        tsdf = rs_tsdf_raw.astype(np.float32) / fd
        rendered = rs_render_raw.astype(np.float32) / fd
        fg = seg > 0

        # Hard source contract checks. Invalid source pixels stay zero.
        expected_rs = np.where(fg & (rendered > 0), rendered,
                        np.where((~fg) & (tsdf > 0), tsdf, 0.)).astype(np.float32)
        if not np.allclose(rs, expected_rs, rtol=0., atol=1e-6):
            raise RuntimeError(f"RealSense rendered-FG/TSDF-BG contract violated at {sid}:{frame}")
        if not np.allclose(gn, gn_render_raw.astype(np.float32)/fd, rtol=0., atol=1e-6):
            raise RuntimeError(f"GN-Trans full-rendered contract violated at {sid}:{frame}")

        record = {
            "scene": sid, "frame": frame,
            "rs_rendered_foreground_path": real.gtdepthpath[ir],
            "rs_tsdf_background_path": real.fusedepthpath[ir],
            "gn_rendered_full_path": trans.gtdepthpath[it],
            "rs_rgb_path": real.colorpath[ir],
            "gn_rgb_path": trans.colorpath[it],
            "object_fraction": float(fg.mean()),
        }
        for name, region in (("all", np.ones_like(fg, dtype=bool)),
                             ("fg", fg), ("bg", ~fg)):
            # The main metrics describe the ACTUAL composed supervision.
            for k, v in _compare(rs, gn, region).items():
                record[f"{name}_{k}"] = v
            # Keep the raw TSDF vs rendered discrepancy as separate evidence.
            for k, v in _compare(tsdf, gn, region).items():
                record[f"raw_tsdf_vs_rendered_{name}_{k}"] = v

        if n < a.loader_audit_pairs:
            r = real.get_data_label(ir)
            t = trans.get_data_label(it)
            crop_eq = np.array_equal(np.asarray(r["crop_box"]), np.asarray(t["crop_box"]))
            k_err = float(np.max(np.abs(np.asarray(r["K"]) - np.asarray(t["K"]))))
            crop_mismatches += int(not crop_eq)
            k_mismatches += int(k_err > 1e-4)
            record["crop_equal"] = bool(crop_eq)
            record["intrinsics_max_abs_diff"] = k_err
            record["cropped_gt_shape_equal"] = bool(np.shape(r["gt_depth_m"]) ==
                                                     np.shape(t["gt_depth_m"]))
            for tag, sample, full_target in (
                ("rs", r, rs), ("gn", t, gn)
            ):
                observed = np.asarray(sample["gt_depth_m"], dtype=np.float32)
                expected = _cropped_target(full_target, sample["crop_box"], observed.shape)
                max_diff = float(np.max(np.abs(observed - expected)))
                record[f"{tag}_loader_gt_max_abs_error_m"] = max_diff
                if not np.isfinite(max_diff) or max_diff > 1e-5:
                    raise RuntimeError(
                        f"{tag} loader target does not match source-composed GT at "
                        f"{sid}:{frame}: max_abs={max_diff:.6g}m")

        rows.append(record)
        if (n + 1) % 100 == 0:
            print(f"[paired-depth] processed {n+1}/{len(schedule)}", flush=True)
    path = Path(a.output_prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No paired frames for diagnostic")
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summary = {
        "contract": DEPTH_CONTRACT,
        "split": a.split, "fraction": a.fraction, "frames": len(rows),
        "loader_audited_pairs": min(len(rows), a.loader_audit_pairs),
        "crop_mismatches": crop_mismatches, "K_mismatches": k_mismatches,
    }
    for region in ("all", "fg", "bg"):
        values = [r[f"{region}_abs_diff_mean_m"] for r in rows
                  if r[f"{region}_abs_diff_mean_m"] is not None]
        summary[f"{region}_mean_abs_difference_m"] = (
            float(np.mean(values)) if values else None)
        ious = [r[f"{region}_valid_iou"] for r in rows
                if r[f"{region}_valid_iou"] is not None]
        summary[f"{region}_mean_valid_iou"] = float(np.mean(ious)) if ious else None
    # Difference of the underlying raw reconstructions (not the training target).
    for region in ("all", "fg", "bg"):
        diffs = [r[f"raw_tsdf_vs_rendered_{region}_abs_diff_mean_m"] for r in rows
                 if r[f"raw_tsdf_vs_rendered_{region}_abs_diff_mean_m"] is not None]
        summary[f"raw_tsdf_vs_rendered_{region}_mean_abs_difference_m"] = (
            float(np.mean(diffs)) if diffs else None)
    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
