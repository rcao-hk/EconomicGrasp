"""Pure-numpy statistics for Height P0-A/B/C; importable without grasp/CUDA.

Conventions:
- All errors use a *fixed, input-defined* reference foreground/ray mask.
- Pixel statistics are exact over that common mask (not model-dependent gating).
- Cluster confidence intervals resample physical scenes, NEVER frame pixels.
- Within-object shape R² centres the GT surface height within one instance/frame.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json

import numpy as np


SCHEMA = "height-mix50-p0abc-v1"
SPLITS = ("train_used", "train_heldout_view", "test_seen", "test_similar", "test_novel")


def deterministic_frames(split, frames_per_scene):
    """Identical frame ids for all checkpoint/appearance comparisons.

    train_used: even frames trained by each MIX50 domain.
    train_heldout_view: odd frames of *same train objects*.
    """
    n = int(frames_per_scene)
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split}")
    if not (1 <= n <= (128 if split.startswith("train_") else 256)):
        raise ValueError("frames_per_scene exceeds allowed unique frame count.")
    if split.startswith("train_"):
        sample = np.rint(np.linspace(0, 127, n)).astype(int)
        return [int(2 * i + (split == "train_heldout_view")) for i in sample]
    sample = np.rint(np.linspace(0, 255, n)).astype(int)
    if len(set(sample.tolist())) != len(sample):
        raise AssertionError("Duplicate evaluation frame selection.")
    return list(map(int, sample))


def scene_ids(split):
    if split in ("train_used", "train_heldout_view"):
        return list(range(100))
    if split == "test_seen":
        return list(range(100, 130))
    if split == "test_similar":
        return list(range(130, 160))
    if split == "test_novel":
        return list(range(160, 190))
    raise ValueError(split)


def stable_seed(parts):
    return int(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16], 16)


def frame_metrics(z, gt, dot, b, seg, height_limits, *, expected_height=None, prior_h=None):
    """Exact masked camera-depth / table-height diagnostics on one 448² frame.

    The mask depends only on reference depth, reference segmentation and
    calibrated ray-plane orientation. It never depends on predicted z.
    """
    for a, name in ((z, "z"), (gt, "gt"), (dot, "dot"), (seg, "seg")):
        if np.asarray(a).ndim != 2:
            raise ValueError(f"{name} must be HxW.")
    z, gt, dot, seg = (np.asarray(x) for x in (z, gt, dot, seg))
    if not (z.shape == gt.shape == dot.shape == seg.shape):
        raise ValueError("Depth/GT/rays/seg resolution mismatch.")
    if not all(np.isfinite(a).all() for a in (z, gt, dot)):
        raise ValueError("Non-finite depth or geometry; no silent deletion.")
    if not (np.isfinite(b) and b > 0):
        raise ValueError("Camera-plane intercept must be a valid positive metre value.")
    h_lo, h_hi = map(float, height_limits)
    if not (-3 < h_lo < h_hi < 3):
        raise ValueError("Invalid explicit height range.")
    fg = (seg > 0) & (gt > 0)
    eligible = dot < -.1
    valid_gt_depth = fg & (gt >= .2) & (gt <= 1.)
    common = valid_gt_depth & eligible
    if common.sum() == 0:
        raise ValueError("No common GT-valid eligible object foreground; audit the camera/table.")
    h_gt = dot * gt + b
    h_pred = dot * z + b
    if expected_height is not None:
        expected_height = np.asarray(expected_height)
        if expected_height.shape != z.shape or not np.isfinite(expected_height).all():
            raise ValueError("Expected decoded predicted height HxW.")
        discrepancy = float(np.max(np.abs(expected_height[common] - h_pred[common])))
        if discrepancy > 1e-4:
            raise ValueError(f"Height/analytic depth inconsistency {discrepancy:.6g}m.")
    diff_z = (z - gt)[common].astype(np.float64)
    diff_h = (h_pred - h_gt)[common].astype(np.float64)
    gt_h = h_gt[common].astype(np.float64)
    pr_h = h_pred[common].astype(np.float64)
    out = {
        "n_pixels": int(common.sum()),
        "n_all_fg_gt": int(fg.sum()),
        "ray_eligible_fraction": float((eligible & fg).sum() / max(fg.sum(), 1)),
        "gt_depth_outside_train_range_fraction": float(
            (fg & ((gt < .2) | (gt > 1.))).sum() / max(fg.sum(), 1)
        ),
        "gt_height_outside_head_range_fraction": float(
            (fg & ((h_gt < h_lo) | (h_gt > h_hi))).sum() / max(fg.sum(), 1)
        ),
        "z_mae_mm": float(np.mean(np.abs(diff_z)) * 1000),
        "z_bias_mm": float(np.mean(diff_z) * 1000),
        "z_p95_abs_mm": float(np.quantile(np.abs(diff_z), .95) * 1000),
        "h_mae_mm": float(np.mean(np.abs(diff_h)) * 1000),
        "h_bias_mm": float(np.mean(diff_h) * 1000),
        "h_p95_abs_mm": float(np.quantile(np.abs(diff_h), .95) * 1000),
        "h_gt_mean_m": float(gt_h.mean()),
        "h_pred_mean_m": float(pr_h.mean()),
        "h_gt_std_mm": float(gt_h.std() * 1000),
        "h_pred_std_mm": float(pr_h.std() * 1000),
        "pred_invalid_depth_fraction": float((z[common] <= 0).mean()),
        # A bounded sigmoid head exists only in the Height family. Reporting
        # a Pose-model h transform near these arbitrary limits as "saturation"
        # would be scientifically wrong.
        "pred_height_near_bound_fraction": (
            float(((pr_h <= h_lo + .02 * (h_hi-h_lo)) |
                   (pr_h >= h_hi - .02 * (h_hi-h_lo))).mean())
            if expected_height is not None else None
        ),
    }
    if expected_height is not None:
        out["height_roundtrip_max_mm"] = discrepancy * 1000
    if prior_h is not None:
        base_z = (float(prior_h) - b) / dot[common]
        base_diff = (base_z - gt[common]).astype(np.float64)
        out["constant_height_z_mae_mm"] = float(np.mean(np.abs(base_diff)) * 1000)
        out["constant_height_h_mae_mm"] = float(np.mean(np.abs(float(prior_h) - gt_h)) * 1000)
        out["learned_minus_constant_z_mae_mm"] = out["z_mae_mm"] - out["constant_height_z_mae_mm"]
    objs = []
    for label in np.unique(seg[common]):
        m = common & (seg == label)
        if not m.any(): continue
        hg = h_gt[m].astype(np.float64)
        hp = h_pred[m].astype(np.float64)
        d = hp - hg
        n = len(hg)
        denom = float(np.sum((hg - hg.mean()) ** 2))
        obj = {
            "label": int(label), "n_pixels": n,
            "z_mae_mm": float(np.mean(np.abs((z-gt)[m]))*1000),
            "h_mae_mm": float(np.mean(np.abs(d))*1000),
            "h_bias_mm": float(np.mean(d)*1000),
            "h_gt_mean_m": float(np.mean(hg)),
            "h_pred_mean_m": float(np.mean(hp)),
            "h_gt_std_mm": float(np.std(hg)*1000),
            "h_pred_std_mm": float(np.std(hp)*1000),
            "within_frame_h_r2": None if denom < 1e-10 else float(1-np.sum(d*d)/denom),
            # Uses this object's own *GT* height median; oracle analysis only,
            # never a deployable baseline or model input.
            "oracle_instance_constant_h_mae_mm": float(
                np.mean(np.abs(hg-np.median(hg)))*1000
            ),
            "h_true_sum": float(hg.sum()),
            "h_true_sum_sq": float(np.sum(hg*hg)),
            "h_pred_sum": float(hp.sum()),
            "h_pred_sum_sq": float(np.sum(hp*hp)),
            "h_err_sum_sq": float(np.sum(d*d)),
            "h_err_abs_sum": float(np.sum(np.abs(d))),
            "h_err_signed_sum": float(np.sum(d)),
        }
        objs.append(obj)
    return out, objs, common, h_pred


def paired_appearance_metrics(z_a, z_b, gt, dot, common, h_a=None, h_b=None):
    """No texture-only causal claim: paired RGBs may also differ in illumination."""
    za, zb, gt, dot = (np.asarray(x) for x in (z_a,z_b,gt,dot))
    for x in (za,zb,gt,dot,common):
        if x.shape!=gt.shape:
            raise ValueError("Paired inputs must share K/crop/mask/GT geometry.")
    if not common.any(): raise ValueError("Empty common RGB appearance comparison.")
    if not np.isfinite(za).all() or not np.isfinite(zb).all():
        raise ValueError("Non-finite paired output.")
    dz = (zb-za)[common].astype(np.float64)
    dh = (dot*(zb-za))[common].astype(np.float64)
    a_err = np.abs(za[common]-gt[common])
    b_err = np.abs(zb[common]-gt[common])
    result = {
        "appearance_z_delta_mae_mm": float(np.mean(np.abs(dz))*1000),
        "appearance_z_delta_p95_mm": float(np.quantile(np.abs(dz),.95)*1000),
        "appearance_h_delta_mae_mm": float(np.mean(np.abs(dh))*1000),
        "appearance_h_delta_p95_mm": float(np.quantile(np.abs(dh),.95)*1000),
        "trans_minus_original_z_mae_mm": float((b_err-a_err).mean()*1000),
        "trans_worse_pixel_fraction": float((b_err > a_err).mean()),
    }
    if h_a is not None and h_b is not None:
        qa,qb=np.asarray(h_a),np.asarray(h_b)
        if qa.shape!=gt.shape or qb.shape!=gt.shape:
            raise ValueError("Height appearance maps differ in shape.")
        result["height_geometry_consistency_max_mm"] = float(
            np.max(np.abs((qb-qa)[common]-dh))*1000)
    return result


def object_pool(rows):
    """Pool object-frame sufficient stats, indexed by (split,scene,instance)."""
    pool = defaultdict(lambda:defaultdict(float))
    frames=defaultdict(set)
    frame_r2=defaultdict(list)
    for row in rows:
        for o in row["objects"]:
            k=(row["family"],row["epoch"],row["split"],row["domain"],
               row["scene"],o["label"])
            for name in ("n_pixels","h_true_sum","h_true_sum_sq","h_pred_sum",
                         "h_pred_sum_sq","h_err_sum_sq","h_err_abs_sum","h_err_signed_sum"):
                pool[k][name]+=o[name]
            frames[k].add(row["frame"])
            if o["within_frame_h_r2"] is not None:
                frame_r2[k].append(float(o["within_frame_h_r2"]))
            pool[k]["oracle_instance_constant_abs_sum_mm"] += (
                o["oracle_instance_constant_h_mae_mm"]*o["n_pixels"]
            )
    out=[]
    for k, a in sorted(pool.items()):
        n=a["n_pixels"]
        gt_var=max(0.,a["h_true_sum_sq"]-a["h_true_sum"]**2/n)
        pred_var=max(0.,a["h_pred_sum_sq"]-a["h_pred_sum"]**2/n)
        out.append(dict(zip(("family","epoch","split","domain","scene","label"),k))|{
            "n_frames":len(frames[k]),"n_pixels":int(n),
            "h_mae_mm":a["h_err_abs_sum"]/n*1000,
            "h_bias_mm":a["h_err_signed_sum"]/n*1000,
            "h_gt_mean_m":a["h_true_sum"]/n,
            "h_pred_mean_m":a["h_pred_sum"]/n,
            "h_gt_std_mm":np.sqrt(gt_var/n)*1000,
            "h_pred_std_mm":np.sqrt(pred_var/n)*1000,
            "h_r2_pooled":None if gt_var<1e-10 else 1-a["h_err_sum_sq"]/gt_var,
            "n_frame_r2_defined":len(frame_r2[k]),
            "median_within_frame_h_r2": (
                float(np.median(frame_r2[k])) if frame_r2[k] else None
            ),
            "positive_within_frame_r2_fraction": (
                float(np.mean(np.asarray(frame_r2[k])>0)) if frame_r2[k] else None
            ),
            "gt_oracle_instance_constant_h_mae_mm": (
                a["oracle_instance_constant_abs_sum_mm"]/n
            ),
        })
    return out


def grouped_ci(rows, metric, *, n_boot=1000, seed=0):
    """Resample physical scene identifiers; equal weight per scene."""
    values=defaultdict(list)
    for r in rows:
        if metric in r and r[metric] is not None:
            values[r["scene"]].append(float(r[metric]))
    if not values:
        return {"mean":None,"ci95":None,"n_scenes":0}
    a=np.asarray([np.mean(v) for _,v in sorted(values.items())])
    rng=np.random.default_rng(seed)
    idx=rng.integers(len(a),size=(int(n_boot),len(a)))
    means=a[idx].mean(1)
    return {"mean":float(a.mean()),"ci95":np.quantile(means,[.025,.975]).tolist(),
            "n_scenes":len(a),"descriptive_small_cluster_count":len(a)<10}



def _scene_metric_values(rows, metric):
    values=defaultdict(list)
    for r in rows:
        if metric in r["metrics"] and r["metrics"][metric] is not None:
            values[r["scene"]].append(float(r["metrics"][metric]))
    return np.asarray([np.mean(v) for _,v in sorted(values.items())])


def _scene_contrast_values(rows):
    values=defaultdict(list)
    for r in rows: values[r["scene"]].append(float(r["delta"]))
    return np.asarray([np.mean(v) for _,v in sorted(values.items())])


def _independent_cluster_gap(a,b,n_boot,seed=0):
    """Gap B - A; independent physical scene clusters for different splits."""
    if len(a)==0 or len(b)==0:return None
    rng=np.random.default_rng(seed)
    ab=a[rng.integers(len(a),size=(n_boot,len(a)))].mean(1)
    bb=b[rng.integers(len(b),size=(n_boot,len(b)))].mean(1)
    return {"gap":float(b.mean()-a.mean()),
            "ci95":np.quantile(bb-ab,[.025,.975]).tolist(),
            "n_train_scenes":len(a),"n_eval_scenes":len(b)}


def generalization_gaps(groups, contrasts, n_boot):
    """Train-to-unseen descriptive gaps; baseline-corrected paired diff-in-gap."""
    metrics=("z_mae_mm","h_mae_mm")
    gaps={}
    family_epoch_domains=sorted(set((f,e,d) for f,e,_,d in groups))
    for family,epoch,domain in family_epoch_domains:
        train=groups.get((family,epoch,"train_used",domain),[])
        for split in ("train_heldout_view","test_seen","test_similar","test_novel"):
            test=groups.get((family,epoch,split,domain),[])
            for metric in metrics:
                gap=_independent_cluster_gap(
                    _scene_metric_values(train,metric),
                    _scene_metric_values(test,metric),n_boot,seed=epoch)
                if gap:
                    gaps[f"{family}/epoch{epoch}/{domain}/{split}/{metric}"]=gap
    # Difference of paired model errors between two independent scene sets:
    # (Height-Pose)_test - (Height-Pose)_train.
    paired={}
    epochs=sorted(set(e for e,_,_,_ in contrasts))
    domains=sorted(set(d for _,_,d,_ in contrasts))
    for e in epochs:
        for domain in domains:
            for metric in metrics:
                train=contrasts.get((e,"train_used",domain,metric),[])
                for split in ("train_heldout_view","test_seen","test_similar","test_novel"):
                    test=contrasts.get((e,split,domain,metric),[])
                    gap=_independent_cluster_gap(
                        _scene_contrast_values(train),
                        _scene_contrast_values(test),n_boot,seed=e)
                    if gap:
                        paired[f"epoch{e}/{domain}/{split}/{metric}"]=gap
    return gaps,paired

def aggregate(rows, *, expected_epochs=None, n_boot=1000):
    """A/B/C summaries and strictly paired, scene-clustered epoch contrasts."""
    seen=set()
    for r in rows:
        k=(r["family"],r["epoch"],r["split"],r["domain"],r["scene"],r["frame"])
        if k in seen: raise ValueError("Duplicate evaluated frame "+str(k))
        seen.add(k)
    groups=defaultdict(list)
    for r in rows:
        groups[(r["family"],r["epoch"],r["split"],r["domain"])].append(r)
    keys=("z_mae_mm","h_mae_mm","z_p95_abs_mm","h_gt_std_mm",
          "h_pred_std_mm","gt_height_outside_head_range_fraction",
          "pred_height_near_bound_fraction","constant_height_z_mae_mm")
    metrics={}
    for k,rs in sorted(groups.items()):
        metrics["/".join(map(str,k))]={
            metric:grouped_ci([{"scene":r["scene"],**r["metrics"]} for r in rs],metric,n_boot=n_boot)
            for metric in keys
        }|{"n_frames":len(rs)}
    family_epochs=defaultdict(set)
    for r in rows: family_epochs[r["family"]].add(int(r["epoch"]))
    missing={family:sorted(set(expected_epochs or [])-present)
             for family,present in family_epochs.items()}
    pairs=defaultdict(dict)
    for r in rows:
        key=(r["epoch"],r["split"],r["domain"],r["scene"],r["frame"])
        pairs[key][r["family"]]=r
    contrasts=defaultdict(list)
    for key,pair in pairs.items():
        if not {"height_analytic","global_film"}<=set(pair):continue
        h,p=pair["height_analytic"],pair["global_film"]
        if h["n_common_pixels"]!=p["n_common_pixels"]:
            raise ValueError("Paired checkpoint masks differ: "+str(key))
        for metric in ("z_mae_mm","h_mae_mm","z_p95_abs_mm"):
            contrasts[(key[0],key[1],key[2],metric)].append({
                "scene":key[3],"delta":h["metrics"][metric]-p["metrics"][metric]
            })
    paired={f"epoch{e}/{split}/{domain}/{metric}":grouped_ci(v,"delta",n_boot=n_boot)
            for (e,split,domain,metric),v in sorted(contrasts.items())}
    gaps, paired_gaps = generalization_gaps(groups,contrasts,n_boot)
    app={}
    for k,rs in sorted(groups.items()):
        pp=[{"scene":r["scene"],**r["paired"]} for r in rs if r.get("paired")]
        if not pp:continue
        app["/".join(map(str,k))]={
            metric:grouped_ci(pp,metric,n_boot=n_boot)
            for metric in ("appearance_z_delta_mae_mm","appearance_h_delta_mae_mm",
                           "trans_minus_original_z_mae_mm","trans_worse_pixel_fraction")
        }|{"n_pairs":len(pp)}
    return {"metrics":metrics,"matched_epoch_contrasts":paired,
            "generalization_gaps":gaps,
            "paired_generalization_gap_differences":paired_gaps,
            "trajectory_eligible": {
                "single_family_at_least_two_epochs": {
                    family:len(epoch_set)>=2 for family,epoch_set in family_epochs.items()
                },
                "matched_both_families_at_least_two_epochs": (
                    len(family_epochs.get("height_analytic",set()) &
                        family_epochs.get("global_film",set())) >= 2
                ),
            },
            "appearance_paired":app,
            "available_epochs":{k:sorted(v) for k,v in family_epochs.items()},
            "missing_expected_epochs":missing,
            "interpretation_limits":{
                "early_stop":"requires retained independent checkpoints; never infer trajectory from e16 alone",
                "view_holdout":"odd frames from same training physical scenes are not unseen objects",
                "appearance":"material/illumination change may include other rendering differences",
                "evaluation":"geometry-only; no official grasp AP or fixed-action counterfactual"
            }}
