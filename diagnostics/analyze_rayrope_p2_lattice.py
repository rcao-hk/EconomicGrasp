#!/usr/bin/env python3
"""Analyze fixed-action RayRoPE P2 lattice experiments.

Reads CSV emitted by run_rayrope_p2_lattice.py. Provides physically matched
per-query winner vs nonwinner, actual-width and insertion sweeps, quality
prefix metrics and scene-clustered paired bootstrap intervals.

This is NOT official GraspNet AP. Each CAD/DexNet label is evaluated for the
executed 17D physical grasp. No training nor label interpolation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from diagnostics.analyze_rayrope_candidates import load_candidates, rank_corr

MUS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)


def mean(a):
    vals = np.asarray(list(a), dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else None


def frame_groups(rows):
    by = defaultdict(list)
    for x in rows:
        by[(x["sample"], x["condition"])].append(x)
    return by


def per_frame_summary(rows, ks=(1, 2, 3, 5, 10, 20)):
    out = []
    for (sample, condition), group in sorted(frame_groups(rows).items()):
        ordered = sorted(group, key=lambda x: (-x["score"], x["candidate"]))
        q = np.asarray([x["quality"] for x in ordered], float)
        if not np.isfinite(q).all():
            raise ValueError(f"Missing exact-action quality in {sample}/{condition}; run evaluator first.")
        item = {
            "sample": sample,
            "scene": ordered[0]["scene"],
            "split": ordered[0]["split"],
            "condition": condition,
            "n": len(group),
            "oracle": float(q.max()),
            "spearman": rank_corr([x["score"] for x in group], [x["quality"] for x in group]),
        }
        for k in ks:
            n = min(k, len(group))
            head = ordered[:n]
            item[f"top{k}_mean_quality"] = float(np.mean([x["quality"] for x in head]))
            item[f"top{k}_success_rate"] = float(np.mean([x["quality"] > 0 for x in head]))
            item[f"top{k}_collision_rate"] = float(np.mean([x["failure"] == "collision" for x in head]))
            item[f"top{k}_empty_rate"] = float(np.mean([x["failure"] == "empty_grasp" for x in head]))
            item[f"top{k}_force_closure_fail_rate"] = float(np.mean([x["failure"] == "force_closure_failure" for x in head]))
        item["top1_regret"] = float(q.max() - q[0])
        for mu in MUS:
            key = f"mu_{mu:.1f}"
            item[f"top1_{key}"] = int(ordered[0]["mu_flags"][mu]) if ordered[0]["mu_flags"][mu] is not None else None
        out.append(item)
    return out


def bootstrap_scene_delta(frame_rows, left, right, metric, *, n_boot=5000, seed=20261106):
    paired = defaultdict(dict)
    for item in frame_rows:
        if item["condition"] in (left, right):
            paired[item["sample"]][item["condition"]] = item
    by_scene = defaultdict(list)
    for sample, pair in paired.items():
        if left not in pair or right not in pair:
            continue
        x, y = pair[left].get(metric), pair[right].get(metric)
        if x is None or y is None or not math.isfinite(float(x)) or not math.isfinite(float(y)):
            continue
        scene = pair[left]["scene"]
        if scene is None:
            raise ValueError(f"Scene ID missing for paired sample {sample!r}")
        by_scene[(pair[left]["split"], scene)].append(float(x) - float(y))
    if not by_scene:
        return {"paired_scenes": 0, "paired_frames": 0, "delta": None, "ci95": None}
    # Three GraspNet object splits have distinct scene populations. Sample
    # scene-clusters *within* each split, then average split means equally.
    by_split = defaultdict(list)
    for (split, _scene), vals in sorted(by_scene.items()):
        by_split[split].append(float(np.mean(vals)))
    splits = sorted(by_split)
    delta = float(np.mean([np.mean(by_split[s]) for s in splits]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        split_draw = []
        for split in splits:
            vals = np.asarray(by_split[split], float)
            split_draw.append(float(np.mean(rng.choice(vals, len(vals), replace=True))))
        boot.append(float(np.mean(split_draw)))
    return {
        "left": left, "right": right, "metric": metric,
        "paired_scenes": len(by_scene), "paired_frames": int(sum(map(len, by_scene.values()))),
        "delta": delta, "ci95": [float(v) for v in np.quantile(boot, [0.025, 0.975])] if n_boot > 0 else None,
        "scope": "fixed_frames_scene_clustered_split_stratified",
    }


def query_winner_pairs(rows):
    by = defaultdict(lambda: defaultdict(list))
    for x in rows:
        if x["condition"] not in ("query_winner", "query_nonwinner"):
            continue
        query = x["raw"].get("query_id", "")
        if query == "":
            raise ValueError("query_id required for query-winner/nonwinner pairing.")
        by[(x["sample"], query)][x["condition"]].append(x)
    details = []
    for (sample, query), pair in sorted(by.items()):
        wins, others = pair["query_winner"], pair["query_nonwinner"]
        if len(wins) != 1 or not others:
            raise ValueError(f"Each sampled query needs one winner and >=1 nonwinner: {sample}/{query}")
        w = wins[0]
        if not math.isfinite(w["quality"]) or not all(math.isfinite(z["quality"]) for z in others):
            raise ValueError("Exact-action quality missing; cannot compute paired query effects.")
        for other in others:
            details.append({
                "sample": sample, "scene": w["scene"], "split": w["split"],
                "query_id": query,
                "winner_quality": float(w["quality"]),
                "nonwinner_quality": float(other["quality"]),
                "quality_delta_winner_minus_nonwinner": float(w["quality"] - other["quality"]),
                "nonwinner_outperforms_winner": bool(other["quality"] > w["quality"]),
                "score_margin": float(w["score"] - other["score"]),
                "winner_failure": w["failure"],
                "nonwinner_failure": other["failure"],
                "same_grasp_action": w["candidate"] == other["candidate"],
            })
    if any(x["same_grasp_action"] for x in details):
        raise RuntimeError("Winner/nonwinner must be distinct physical actions")
    return details


def sweep_families(rows, condition):
    by = defaultdict(list)
    for x in rows:
        if x["condition"] == condition:
            parent = x["raw"].get("parent_id", "")
            if not parent:
                raise ValueError(f"parent_id missing in sweep {condition}")
            by[(x["sample"], parent)].append(x)
    details = []
    for (sample, parent), group in sorted(by.items()):
        if len(group) < 2:
            continue
        if any(not math.isfinite(x["quality"]) for x in group):
            raise ValueError("Missing exact-action quality in sweep")
        physical = [x["candidate"] for x in group]
        if len(physical) != len(set(physical)):
            raise ValueError(f"Duplicate physical action in sweep {sample}/{parent}")
        q = np.asarray([x["quality"] for x in group])
        s = np.asarray([x["score"] for x in group])
        best = int(q.argmax())
        pred = next((i for i, x in enumerate(group) if x["raw"].get("width_mode") == "predicted"), None)
        result = {
            "sample": sample, "scene": group[0]["scene"], "split": group[0]["split"],
            "parent_id": parent, "condition": condition, "n": len(group),
            "quality_range": float(np.ptp(q)),
            "oracle_quality": float(q.max()),
            "best_action_id": physical[best],
            "score_std": float(s.std()),
            "score_quality_spearman": rank_corr(s, q),
        }
        if condition == "width_sweep":
            depths = [x["insertion"] for x in group]
            centers = [x["center"] for x in group]
            if np.ptp(depths) > 1e-7 or any(not np.allclose(c, centers[0], atol=1e-6) for c in centers):
                raise ValueError(f"Width sweep changed insertion or center: {sample}/{parent}")
            result["best_width_m"] = float(group[best]["width_m"])
            result["predicted_width_m"] = float(group[pred]["width_m"]) if pred is not None else None
            result["predicted_quality"] = float(q[pred]) if pred is not None else None
            result["regret_predicted_width"] = float(q.max()-q[pred]) if pred is not None else None
        if condition.startswith("insertion_sweep"):
            depths = [x["insertion"] for x in group]
            result["best_insertion_m"] = float(group[best]["insertion"])
            result["rank_depth_quality"] = rank_corr(depths, q)
            result["score_selected_quality"] = float(q[np.argmax(s)])
            result["regret_score_selected"] = float(q.max()-q[np.argmax(s)])
            if condition == "insertion_sweep_fixed_width" and np.ptp([x["width_m"] for x in group]) > 1e-6:
                raise ValueError(f"Fixed-width insertion sweep changed width: {sample}/{parent}")
        details.append(result)
    return details


def summarize_sweep(rows):
    return {
        "num_families": len(rows),
        "mean_quality_range": mean([x["quality_range"] for x in rows]),
        "fraction_quality_changes": mean([float(x["quality_range"] > 1e-8) for x in rows]),
        "mean_oracle_quality": mean([x["oracle_quality"] for x in rows]),
        "mean_regret_predicted_width": mean([x.get("regret_predicted_width") for x in rows if x.get("regret_predicted_width") is not None]),
        "mean_regret_score_selected_depth": mean([x.get("regret_score_selected") for x in rows if x.get("regret_score_selected") is not None]),
        "mean_score_std": mean([x["score_std"] for x in rows]),
        "note": "All oracle values are conditional on evaluated finite width/depth grids.",
    }


def _dump(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            return float(x) if math.isfinite(float(x)) else None
        return x
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix+".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    rows, schema = load_candidates(args.input_csv, None, "m")
    if any(not math.isfinite(x["quality"]) for x in rows):
        raise ValueError("Missing exact-action labels. Run P2 runner without --diag_skip_eval.")
    frame = per_frame_summary(rows)
    paired = query_winner_pairs(rows)
    sweeps = {
        condition: sweep_families(rows, condition)
        for condition in ("width_sweep", "insertion_sweep_fixed_width", "insertion_sweep_predicted_width")
    }
    pair_per_sample = defaultdict(list)
    for x in paired:
        pair_per_sample[x["sample"]].append(x)
    pair_frames = []
    for sample, pairs in sorted(pair_per_sample.items()):
        pair_frames.append({
            "sample": sample, "scene": pairs[0]["scene"], "split": pairs[0]["split"],
            "condition": "paired_query",
            "mean_winner_minus_nonwinner_quality": float(np.mean([x["quality_delta_winner_minus_nonwinner"] for x in pairs])),
            "nonwinner_better_fraction": float(np.mean([x["nonwinner_outperforms_winner"] for x in pairs])),
            "n_pairs": len(pairs),
        })

    overall = defaultdict(dict)
    conditions = sorted({x["condition"] for x in frame})
    keys = ["top1_mean_quality", "top5_mean_quality", "top1_regret", "spearman", "top5_collision_rate", "top5_empty_rate", "top5_force_closure_fail_rate"]
    for name in conditions:
        subset = [x for x in frame if x["condition"] == name]
        overall[name] = {
            "frames": len(subset), "scenes": len({x["scene"] for x in subset}),
            **{k: mean([x[k] for x in subset]) for k in keys},
        }

    contrasts = {}
    for left, right in (("query_winner", "query_nonwinner"), ("query_winner", "lattice_uniform")):
        if left in conditions and right in conditions:
            contrasts[f"{left}_minus_{right}"] = {
                key: bootstrap_scene_delta(frame, left, right, key, n_boot=args.n_boot, seed=args.seed)
                for key in keys
            }
    result = {
        "input": args.input_csv, "schema": schema,
        "frames": len({x["sample"] for x in rows}),
        "scenes": len({x["scene"] for x in rows if x["scene"] is not None}),
        "overall_by_condition": overall,
        "paired_scene_contrasts": contrasts,
        "query_winner_nonwinner": {
            "pairs": len(paired),
            "mean_winner_quality_gain": mean([x["quality_delta_winner_minus_nonwinner"] for x in paired]),
            "nonwinner_better_fraction": mean([float(x["nonwinner_outperforms_winner"]) for x in paired]),
            "per_frame": pair_frames,
            "per_query": paired,
        },
        "width_sweep": {
            **summarize_sweep(sweeps["width_sweep"]),
            "families": sweeps["width_sweep"],
        },
        "insertion_sweep_fixed_width": {
            **summarize_sweep(sweeps["insertion_sweep_fixed_width"]),
            "families": sweeps["insertion_sweep_fixed_width"],
        },
        "insertion_sweep_predicted_width": {
            **summarize_sweep(sweeps["insertion_sweep_predicted_width"]),
            "families": sweeps["insertion_sweep_predicted_width"],
        },
        "per_frame_condition": frame,
        "interpretation": [
            "Query-winner and nonwinner candidates are sampled from the same native top-view and matched query anchors; the physical actions differ.",
            "Fixed-width insertion sweeps hold center, rotation, width fixed; insertion varies and model depth-bin utility changes.",
            "Width sweeps hold center, rotation, insertion and model score fixed; executed physical width changes.",
            "Full lattice uniform and per-query argmax winner are different candidate populations. Their difference is not a pure sampler causal effect.",
            "Paired scene bootstrap does not capture training-seed variance; these are offline exact CAD/DexNet labels, not robot success or official GraspNet AP.",
        ],
    }
    _dump(args.output_json, result)
    print(json.dumps({"output": args.output_json, "frames": result["frames"], "query_pairs": len(paired),
                      "width_families": len(sweeps["width_sweep"]),
                      "insertion_families": len(sweeps["insertion_sweep_fixed_width"])}, indent=2))


if __name__ == "__main__":
    main()
