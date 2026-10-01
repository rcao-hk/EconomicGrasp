#!/usr/bin/env python3
"""Compare local-selection x global-scoring 2x2 official GraspNet results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CONDITIONS = (
    "lbase_gbase",
    "lfull_gbase",
    "lbase_gfull",
    "lfull_gfull",
)
SPLITS = ("test_seen", "test_similar", "test_novel")
LABELS = {
    "lbase_gbase": "Local Base / Global Base",
    "lfull_gbase": "Local Full / Global Base",
    "lbase_gfull": "Local Base / Global Full",
    "lfull_gfull": "Local Full / Global Full",
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--collision", choices=("on", "off", "both"), default="both")
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    return p


def scalar_ap(value):
    while isinstance(value, list):
        if not value:
            raise ValueError("Empty reported_ap")
        value = value[0]
    return float(value)


def load_one(root, cond, mode, split):
    d = root / cond / f"test_collision_{mode}" / "official" / split
    summary = json.loads((d / "summary.json").read_text())
    acc = np.load(d / "accuracy.npy", allow_pickle=False)
    if acc.ndim != 4 or acc.shape[0] != 30 or acc.shape[-2:] != (50, 6):
        raise RuntimeError(f"Malformed accuracy array: {d} {acc.shape}")
    if not np.isfinite(acc).all():
        raise RuntimeError(f"Non-finite accuracy: {d}")
    ap = scalar_ap(summary["reported_ap"])
    if abs(ap - float(acc.mean())) > 1e-7:
        raise RuntimeError(f"summary/accuracy mismatch: {d}")
    protocol_path = (
        root / cond / f"test_collision_{mode}" / split / "protocol.json"
    )
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("local_action_source") not in ("base", "full"):
        raise RuntimeError(f"Missing local-action protocol: {protocol_path}")
    if protocol.get("global_score_source") not in ("base", "full"):
        raise RuntimeError(f"Missing global-score protocol: {protocol_path}")
    return summary, acc, protocol


def paired_bootstrap(delta_scene, n, seed):
    delta_scene = np.asarray(delta_scene, dtype=np.float64)
    if delta_scene.shape != (30,):
        raise ValueError(delta_scene.shape)
    rng = np.random.default_rng(seed)
    if n <= 0:
        return None
    # Chunk to keep memory bounded.
    vals = []
    left = int(n)
    while left:
        k = min(left, 4096)
        ids = rng.integers(0, 30, size=(k, 30))
        vals.append(delta_scene[ids].mean(axis=1))
        left -= k
    boot = np.concatenate(vals)
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def main():
    a = parser().parse_args()
    if a.bootstrap < 0:
        raise ValueError("--bootstrap must be >=0")
    root = Path(a.root)
    modes = ("on", "off") if a.collision == "both" else (a.collision,)
    output = {
        "primary_collision": "on",
        "conditions": LABELS,
        "modes": {},
        "notes": [
            "This is a descriptive 2x2 intervention, not an additive causal decomposition of AP.",
            "Local source chooses angle-depth. Global source scores that exact chosen physical action.",
            "Centre, view and frozen source width map are unchanged.",
            "Collision-on is the primary benchmark result; collision-off is secondary mechanism evidence.",
        ],
    }

    for mode_i, mode in enumerate(modes):
        data = {}
        protocol_ref = None
        for cond in CONDITIONS:
            data[cond] = {}
            for split in SPLITS:
                summary, acc, protocol = load_one(root, cond, mode, split)
                expected_local = "full" if cond.startswith("lfull") else "base"
                expected_global = "full" if cond.endswith("gfull") else "base"
                if protocol["local_action_source"] != expected_local:
                    raise RuntimeError(f"{cond}: wrong local source")
                if protocol["global_score_source"] != expected_global:
                    raise RuntimeError(f"{cond}: wrong global source")

                invariant = {
                    k: v
                    for k, v in protocol.items()
                    if k
                    not in (
                        "score_source",
                        "local_action_source",
                        "global_score_source",
                        "intervention",
                    )
                }
                if protocol_ref is None:
                    protocol_ref = invariant
                else:
                    # Collision mode/split naturally differ; compare important
                    # paired provenance explicitly instead of entire JSON.
                    for key in (
                        "checkpoint_sha256",
                        "source_checkpoint_epoch",
                        "full_control_epoch",
                        "full_control_sha256",
                        "evaluation_fraction",
                        "primary_collision",
                    ):
                        if invariant.get(key) != protocol_ref.get(key):
                            raise RuntimeError(
                                f"Protocol mismatch for {key}: {cond}/{split}"
                            )

                data[cond][split] = {
                    "ap": scalar_ap(summary["reported_ap"]),
                    "scene": acc.mean(axis=(1, 2, 3)).astype(np.float64),
                }

        contrasts = {
            "local_at_base_global": ("lfull_gbase", "lbase_gbase"),
            "global_at_base_local": ("lbase_gfull", "lbase_gbase"),
            "joint_full_vs_base": ("lfull_gfull", "lbase_gbase"),
            "local_at_full_global": ("lfull_gfull", "lbase_gfull"),
            "global_at_full_local": ("lfull_gfull", "lfull_gbase"),
        }

        mode_result = {
            "ap": {
                cond: {split: data[cond][split]["ap"] for split in SPLITS}
                for cond in CONDITIONS
            },
            "contrasts": {},
            "interaction": {},
        }
        for cond in CONDITIONS:
            vals = [mode_result["ap"][cond][s] for s in SPLITS]
            mode_result["ap"][cond]["mean"] = float(np.mean(vals))

        for contrast_i, (name, (hi, lo)) in enumerate(contrasts.items()):
            mode_result["contrasts"][name] = {}
            for split_i, split in enumerate(SPLITS):
                delta_scene = data[hi][split]["scene"] - data[lo][split]["scene"]
                mode_result["contrasts"][name][split] = {
                    "delta_ap": float(delta_scene.mean()),
                    "scene_positive_fraction": float((delta_scene > 0).mean()),
                    "scene_median_delta": float(np.median(delta_scene)),
                    "scene_bootstrap95": paired_bootstrap(
                        delta_scene,
                        a.bootstrap,
                        a.seed + 10000 * mode_i + 1000 * contrast_i + split_i,
                    ),
                }

        # Descriptive two-factor interaction:
        # FF - FB - BF + BB. AP is nonlinear, so do not interpret this as a
        # strict causal interaction or expect exact additive decomposition.
        for split_i, split in enumerate(SPLITS):
            scene = (
                data["lfull_gfull"][split]["scene"]
                - data["lfull_gbase"][split]["scene"]
                - data["lbase_gfull"][split]["scene"]
                + data["lbase_gbase"][split]["scene"]
            )
            mode_result["interaction"][split] = {
                "delta_ap": float(scene.mean()),
                "scene_bootstrap95": paired_bootstrap(
                    scene,
                    a.bootstrap,
                    a.seed + 90000 + 10000 * mode_i + split_i,
                ),
            }
        output["modes"][mode] = mode_result

    out = root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Local Selection × Global Scoring 2×2",
        "",
        "**Primary reporting: collision-on.**",
        "",
    ]
    for mode in modes:
        r = output["modes"][mode]
        lines += [
            f"## Collision {mode}",
            "",
            "| Local selector | Global scorer | Seen AP | Similar AP | Novel AP | Mean |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for cond in CONDITIONS:
            ap = r["ap"][cond]
            local = "Full" if cond.startswith("lfull") else "Base"
            glob = "Full" if cond.endswith("gfull") else "Base"
            lines.append(
                f"| {local} | {glob} | {100*ap['test_seen']:.2f} | "
                f"{100*ap['test_similar']:.2f} | {100*ap['test_novel']:.2f} | "
                f"{100*ap['mean']:.2f} |"
            )
        lines += [
            "",
            "| Contrast | Seen ΔAP | Similar ΔAP | Novel ΔAP |",
            "|---|---:|---:|---:|",
        ]
        pretty = {
            "local_at_base_global": "Local: Base→Full, Global=Base",
            "global_at_base_local": "Global: Base→Full, Local=Base",
            "joint_full_vs_base": "Joint: BB→FF",
            "local_at_full_global": "Local: Base→Full, Global=Full",
            "global_at_full_local": "Global: Base→Full, Local=Full",
        }
        for name, label in pretty.items():
            x = r["contrasts"][name]
            lines.append(
                f"| {label} | {100*x['test_seen']['delta_ap']:+.2f} | "
                f"{100*x['test_similar']['delta_ap']:+.2f} | "
                f"{100*x['test_novel']['delta_ap']:+.2f} |"
            )
        inter = r["interaction"]
        lines.append(
            f"| Descriptive 2×2 interaction | "
            f"{100*inter['test_seen']['delta_ap']:+.2f} | "
            f"{100*inter['test_similar']['delta_ap']:+.2f} | "
            f"{100*inter['test_novel']['delta_ap']:+.2f} |"
        )
        lines += [""]

    lines += [
        "The interaction is descriptive only: GraspNet AP and collision filtering "
        "are nonlinear, so the table is not an additive causal decomposition.",
        "",
    ]
    (out / "comparison.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
