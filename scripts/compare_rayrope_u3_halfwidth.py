#!/usr/bin/env python3
"""Compare official GraspNet AP across frozen-U3 RayRoPE halfwidth interventions."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

VALID = {"learned", "point", "constant_train", "fixed20", "shuffle"}
SPLITS = {"test_seen", "test_similar", "test_novel"}


def _load(root, mode, split, collision):
    target = root / mode / f"test_collision_{collision}"
    proto = json.loads((target / split / "protocol.json").read_text())
    official = target / "official" / split
    summary = json.loads((official / "summary.json").read_text())
    a = np.load(official / "accuracy.npy", allow_pickle=False)
    if a.shape != (30, 26, 50, 6) or not np.isfinite(a).all():
        raise ValueError(f"Bad official AP tensor for {mode}/{split}/{collision}")
    if proto["diagnostic"]["mode"] != mode or proto["batch_size"] != 3:
        raise ValueError("Wrong diagnostic mode or inference batch")
    if len(proto["schedule"]) != 780 or proto["max_frames"]:
        raise ValueError("Refuse partial diagnostic")
    if abs(float(a.mean()) - float(summary["mean_accuracy"])) > 1e-9:
        raise ValueError("Official summary does not match accuracy.npy")
    return a, proto, target


def _same_contract(a, b):
    keep = ("checkpoint_sha256", "code_sha256", "training_protocol",
            "checkpoint_epoch", "split", "schedule", "batch_size",
            "num_shards", "collision_filter", "depth_bias_mm",
            "dataset_root", "evaluation_fraction", "seed")
    for k in keep:
        if a[k] != b[k]:
            raise ValueError(f"Confounded diagnostic comparison: {k}")
    da, db = a["diagnostic"], b["diagnostic"]
    for k in ("script_sha256", "train_metrics_sha256",
              "constant_train_halfwidth_m", "shuffle_seed"):
        if da[k] != db[k]:
            raise ValueError(f"Different diagnostic configuration: {k}")


def _signature(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _proposal_hashes(target, proto, split):
    result = {}
    digest = _signature(proto)
    for scene, anno in proto["schedule"]:
        marker = target / split / "completed" / f"{scene:04d}_{anno:04d}.json"
        row = json.loads(marker.read_text())
        if row["signature"] != digest:
            raise ValueError(f"Wrong per-frame signature at {marker}")
        h = row.get("proposal_sha256")
        if not isinstance(h, str) or len(h) != 64:
            raise ValueError(f"Missing upstream proposal SHA at {marker}")
        result[(scene, anno)] = h
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--modes", default="learned point constant_train fixed20 shuffle")
    p.add_argument("--splits", default="test_seen test_similar test_novel")
    p.add_argument("--bootstrap", type=int, default=2000)
    a = p.parse_args()
    modes, splits = a.modes.replace(",", " ").split(), a.splits.replace(",", " ").split()
    if not modes or "learned" not in modes or len(modes) != len(set(modes)):
        raise ValueError("Provide unique diagnostic modes including learned")
    if set(modes) - VALID or set(splits) - SPLITS or not splits:
        raise ValueError("Unknown mode or split")
    if a.bootstrap < 0:
        raise ValueError("bootstrap must be >=0")
    root = Path(a.root).resolve()
    random = np.random.default_rng(20261013)
    results, deltas = [], {}
    checkpoint_sha = None
    for collision in ("on", "off"):
        for split in splits:
            reference, ref_proto, ref_root = _load(root, "learned", split, collision)
            if checkpoint_sha is None:
                checkpoint_sha = ref_proto["checkpoint_sha256"]
            elif checkpoint_sha != ref_proto["checkpoint_sha256"]:
                raise ValueError("Different U3 checkpoints across splits/collision modes")
            ref_proposals = _proposal_hashes(ref_root, ref_proto, split)
            for mode in modes:
                arr, proto, folder = _load(root, mode, split, collision)
                _same_contract(ref_proto, proto)
                if _proposal_hashes(folder, proto, split) != ref_proposals:
                    raise AssertionError(f"Upstream proposals changed in {mode}/{split}/{collision}")
                scene_diff = (arr - reference).mean(axis=(1, 2, 3)) * 100
                deltas[(mode, collision, split)] = scene_diff
                if a.bootstrap:
                    sample = random.integers(0, len(scene_diff), (a.bootstrap, len(scene_diff)))
                    ci = np.quantile(scene_diff[sample].mean(axis=1), [.025, .975])
                    lo, hi = float(ci[0]), float(ci[1])
                else:
                    lo = hi = None
                results.append(dict(mode=mode, collision=collision, split=split,
                    ap_pct=float(arr.mean() * 100), delta_vs_learned_pp=float(scene_diff.mean()),
                    ci95_low_pp=lo, ci95_high_pp=hi, identical_proposals=True,
                    frames=780, checkpoint_sha256=checkpoint_sha))
    if set(splits) == SPLITS:
        for collision in ("on", "off"):
            for mode in modes:
                rows = [x for x in results if x["mode"] == mode and x["collision"] == collision]
                values = np.concatenate([deltas[(mode, collision, split)] for split in sorted(SPLITS)])
                if a.bootstrap:
                    sample = random.integers(0, len(values), (a.bootstrap, len(values)))
                    ci = np.quantile(values[sample].mean(axis=1), [.025, .975])
                    lo, hi = float(ci[0]), float(ci[1])
                else:
                    lo = hi = None
                results.append(dict(mode=mode, collision=collision, split="mean",
                    ap_pct=float(np.mean([r["ap_pct"] for r in rows])),
                    delta_vs_learned_pp=float(values.mean()),
                    ci95_low_pp=lo, ci95_high_pp=hi, identical_proposals=True,
                    frames=2340, checkpoint_sha256=checkpoint_sha))
    output = root / "comparison"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    output.with_suffix(".json").write_text(json.dumps(dict(
        note="Frozen U3: only RayRoPE halfwidth changes; scene CI is not multi-seed variance",
        rows=results), indent=2))
    md = ["# Frozen-U3 RayRoPE halfwidth diagnostic", "",
          "| Mode | Collision | Split | AP (%) | Δ vs learned (pp) |",
          "|---|---|---|---:|---:|"]
    for row in results:
        md.append(f"| {row['mode']} | {row['collision']} | {row['split']} | "
                  f"{row['ap_pct']:.4f} | {row['delta_vs_learned_pp']:+.4f} |")
    md += ["", "All conditions share the same U3 checkpoint, RGB frames and proposals.",
           "Comparison is a frozen-checkpoint grouping intervention, not a training ablation.",
           "Scene-bootstrap intervals in CSV/JSON do not measure training-seed variability."]
    output.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"[U3-DIAG] reports at {output}.{{csv,json,md}}")
    for row in results:
        if row["split"] == "mean":
            print(f"  {row['mode']} collision-{row['collision']}: "
                  f"{row['ap_pct']:.4f}% ({row['delta_vs_learned_pp']:+.4f}pp)")


if __name__ == "__main__":
    main()
