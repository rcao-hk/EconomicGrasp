"""Analyze full candidate conflict graphs and exact-quality selection oracles."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .core import (
    SCHEMA, component_fixed_cardinality_oracle, digest, frame_key,
    greedy_nms, official_ap_matrix, physical_ids, q_priority,
    scene_cluster_bootstrap, sha256_file, write_json, write_npz,
)
from .label import load_verified_manifest
from .build import load_verified_manifest as load_graph_manifest


VARIANTS = (
    "native",
    "quality_nms_native_rank",
    "quality_rank_native_nms",
    "quality_nms_quality_rank",
    "component_set_native_rank",
    "component_set_quality_rank",
    "per_mu_success_oracle",
)


def _load_label(labels_root: Path, record: dict, graph_manifest_sha: str):
    path = labels_root / record["labels_path"]
    if not path.is_file() or sha256_file(path) != record["labels_sha256"]:
        raise RuntimeError(f"Missing/corrupt exact-action labels: {path}")
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("contract", {}).get("graph_manifest_sha256") != graph_manifest_sha:
        raise RuntimeError(f"Label cache graph provenance mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    if not np.array_equal(physical_ids(out["grasps"]), out["physical_id"].astype(str)):
        raise RuntimeError(f"Label cache physical identity mismatch: {path}")
    if not np.array_equal(out["success_bits"].sum(axis=1), out["quality_count"]):
        raise RuntimeError(f"Quality count/threshold bits mismatch: {path}")
    return out


def _frame_ap(
    grasps, assigned, bad, friction, bits, edges, component_id,
    native_keep, suppressor, max_exact_component_nodes,
):
    native_scores = np.asarray(grasps[:, 0], dtype=np.float64)
    q_count = np.asarray(bits, dtype=np.uint8).sum(axis=1).astype(np.int64)
    qscore = q_priority(q_count, native_scores)
    n = len(grasps)

    ap = {}
    ap["native"] = official_ap_matrix(
        grasps, assigned, bad, friction, native_keep, native_scores
    )

    q_keep, _, _ = greedy_nms(qscore, edges)
    ap["quality_nms_native_rank"] = official_ap_matrix(
        grasps, assigned, bad, friction, q_keep, native_scores
    )

    ap["quality_rank_native_nms"] = official_ap_matrix(
        grasps, assigned, bad, friction, native_keep, qscore
    )

    ap["quality_nms_quality_rank"] = official_ap_matrix(
        grasps, assigned, bad, friction, q_keep, qscore
    )

    component_keep, component_info = component_fixed_cardinality_oracle(
        n, edges, component_id, native_keep, q_count, native_scores,
        max_component_nodes=max_exact_component_nodes,
    )
    ap["component_set_native_rank"] = official_ap_matrix(
        grasps, assigned, bad, friction, component_keep, native_scores
    )
    ap["component_set_quality_rank"] = official_ap_matrix(
        grasps, assigned, bad, friction, component_keep, qscore
    )

    # Six separate optimistic policies: for each friction threshold, the oracle
    # knows which candidates truly succeed at that threshold. This is not a
    # deployable scalar ranker and is not a single shared candidate ordering.
    per_mu = np.zeros((50, 6), dtype=np.float64)
    for j in range(6):
        priority = bits[:, j].astype(np.float64) + 1e-6 * (
            (native_scores - native_scores.min()) /
            max(float(native_scores.max() - native_scores.min()), 1e-12)
        ) if n else native_scores.copy()
        keep_mu, _, _ = greedy_nms(priority, edges)
        per_mu[:, j] = official_ap_matrix(
            grasps, assigned, bad, friction, keep_mu, priority
        )[:, j]
    ap["per_mu_success_oracle"] = per_mu

    suppressed = np.flatnonzero(np.asarray(suppressor) >= 0)
    better = suppressed[q_count[suppressed] > q_count[suppressor[suppressed]]]
    return ap, {
        "native_nms_survivors": int(len(native_keep)),
        "quality_nms_survivors": int(len(q_keep)),
        "quality_nms_changed_nodes": int(len(set(native_keep.tolist()) ^ set(q_keep.tolist()))),
        "native_selected_q_sum": int(q_count[native_keep].sum()),
        "quality_nms_selected_q_sum": int(q_count[q_keep].sum()),
        "component_selected_q_sum": int(q_count[component_keep].sum()),
        "component_oracle": component_info,
        "suppressed_candidates": int(len(suppressed)),
        "suppressed_candidates_with_higher_q_than_suppressor": int(len(better)),
        "fraction_suppressed_better": float(len(better) / len(suppressed)) if len(suppressed) else 0.0,
        "mean_q_gap_better_suppressed": float(
            np.mean(q_count[better] - q_count[suppressor[better]])
        ) if len(better) else 0.0,
    }


def _mean_ap(ap: np.ndarray) -> float:
    return float(np.asarray(ap, dtype=np.float64).mean() * 100.0)


def _bootstrap_delta(scene_ap: dict[int, np.ndarray], variant: str,
                     n_boot: int, seed: int) -> dict:
    scenes = sorted(scene_ap)
    deltas = {}
    for scene in scenes:
        values = scene_ap[scene]
        deltas[scene] = float((values[variant] - values["native"]) * 100.0)
    return scene_cluster_bootstrap(deltas, n_boot=n_boot, seed=seed)


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(summary: dict) -> str:
    lines = [
        f"# Candidate Conflict Graph + Quality-aware Selection Oracle — {summary['split']}",
        "",
        f"Frames: {summary['num_frames']} / expected {summary['expected_frames']}; "
        f"methods: {', '.join(summary['methods'])}.",
        "",
        "> Conflict edges reproduce the GraspNet NMS geometric suppression relation; "
        "they do not prove physical incompatibility. Oracle policies use exact CAD/Dex-Net "
        "labels and are diagnostic upper bounds, not deployable methods.",
        "",
        "| Method | Native AP | Q-NMS / native rank | Q-rank / native NMS | Q-NMS + Q-rank | Component set / native rank | Per-μ oracle |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, result in summary["methods"].items():
        m = result["mean_ap_percent"]
        lines.append(
            f"| {method} | {m['native']:.4f} | {m['quality_nms_native_rank']:.4f} | "
            f"{m['quality_rank_native_nms']:.4f} | {m['quality_nms_quality_rank']:.4f} | "
            f"{m['component_set_native_rank']:.4f} | {m['per_mu_success_oracle']:.4f} |"
        )
    lines += ["", "## Oracle gains over native", "",
              "| Method | Oracle | Δ AP (pp) | Scene-bootstrap 95% CI (pp) |", "|---|---|---:|---:|"]
    for method, result in summary["methods"].items():
        for variant, boot in result["scene_bootstrap"].items():
            lines.append(
                f"| {method} | {variant} | {boot['mean']:.4f} | "
                f"[{boot['ci95'][0]:.4f}, {boot['ci95'][1]:.4f}] |"
            )
    lines += ["", "## Official AP parity", ""]
    for method, result in summary["methods"].items():
        parity = result["official_ap_parity"]
        lines.append(
            f"- **{method}**: max_abs={parity['max_abs']:.8g}, "
            f"mean_abs={parity['mean_abs']:.8g}, passed={parity['passed']}"
        )
    lines += ["", "## Graph and suppression diagnostics", "",
              "| Method | Candidates | Edges | Components | NMS survivors | Suppressed Q-better than suppressor | Component exact node coverage |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for method, result in summary["methods"].items():
        g = result["graph_totals"]
        s = result["selection_totals"]
        lines.append(
            f"| {method} | {g['nodes']} | {g['edges']} | {g['components']} | "
            f"{g['native_survivors']} | {s['suppressed_q_better']} | "
            f"{s['component_exact_node_coverage']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation boundaries",
        "",
        "- Q-NMS uses mean success count across six friction thresholds; it is not an exact AP optimizer.",
        "- Per-μ oracle uses the true success bit for that threshold and is an intentionally optimistic, threshold-specific policy.",
        "- Component-set oracle maximizes total Q while preserving the baseline NMS survivor count per connected component. Components above the exact-size limit fall back to native NMS; report exact-node coverage.",
        "- A high oracle gain means selection has headroom under the current candidate pool and labels; it does not prove a learned selector can attain it.",
        "- All confidence intervals resample scenes, not individual frames/candidates, and do not include training-seed variance.",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-root", required=True)
    p.add_argument("--labels-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--official-ap", action="append", default=[],
                   help="Optional/required formal mapping NAME=/path/ap_split_camera.npy")
    p.add_argument("--max-exact-component-nodes", type=int, default=20)
    p.add_argument("--ap-tolerance", type=float, default=1e-5)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-partial", action="store_true", help="Smoke only")
    p.add_argument("--skip-official-parity", action="store_true", help="Smoke only; formal runs must compare official AP")
    a = p.parse_args()

    graph_root = Path(a.graph_root)
    labels_root = Path(a.labels_root)
    gm = load_graph_manifest(graph_root / "graph_manifest.json")
    lm = load_verified_manifest(labels_root / "label_manifest.json")
    if lm.get("graph_manifest_sha256") != gm["manifest_sha256"]:
        raise RuntimeError("Graph/label manifests are not the same immutable experiment")
    if not gm.get("complete") or not lm.get("complete"):
        if not a.allow_partial:
            raise RuntimeError("Formal analysis requires complete 100% graph and label manifests")
    if not a.skip_official_parity and not gm.get("complete"):
        raise RuntimeError("Official AP parity cannot be skipped for a formal complete run")
    if a.max_exact_component_nodes < 1:
        raise ValueError("--max-exact-component-nodes must be positive")

    ap_paths = {}
    for item in a.official_ap:
        if "=" not in item:
            raise ValueError("--official-ap must use NAME=PATH")
        name, path = item.split("=", 1)
        if name in ap_paths:
            raise ValueError(f"Duplicate official AP for {name}")
        ap_paths[name] = Path(path)
    out_root = Path(a.output)
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Analysis output must be empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    # Label sidecar records are indexed by frame key from the per-scene manifest.
    label_records = {}
    for scene in lm["scene_records"]:
        for record in scene["frames"]:
            label_records[record["contract"]["frame_key"]] = record

    summary = {
        "schema": SCHEMA,
        "split": gm["split"],
        "camera": gm["camera"],
        "num_frames": len(next(iter(gm["methods"].values()))["entries"]),
        "expected_frames": gm["protocol"]["expected_full_frames"],
        "complete": bool(gm["complete"] and lm["complete"]),
        "methods": {},
    }
    frame_rows = []
    scene_rows = []

    for method, method_manifest in gm["methods"].items():
        official_path = ap_paths.get(method)
        if official_path is None and method_manifest.get("official_ap_path"):
            official_path = Path(method_manifest["official_ap_path"])
        official = None
        if official_path is not None:
            if not official_path.is_file():
                raise FileNotFoundError(f"Official AP file missing for {method}: {official_path}")
            official = np.asarray(np.load(official_path, allow_pickle=False), dtype=np.float64)
            if official.shape != (30, 256, 50, 6) or not np.isfinite(official).all():
                raise ValueError(f"{method}: expected official AP [30,256,50,6], got {official.shape}")
        if official is None and not a.skip_official_parity:
            raise ValueError(f"Formal parity requires --official-ap for {method}")

        entries = sorted(method_manifest["entries"], key=lambda x: (x["scene"], x["anno"]))
        frame_ap = {v: [] for v in VARIANTS}
        per_frame_rows = []
        per_scene_acc = {}
        graph_totals = dict(nodes=0, edges=0, components=0, native_survivors=0)
        selection_totals = dict(
            suppressed=0, suppressed_q_better=0, component_exact_nodes=0,
            component_total_nodes=0, component_exact_components=0, component_fallback_components=0,
        )
        parity_errors = []

        for frame_number, entry in enumerate(entries, 1):
            scene, anno = int(entry["scene"]), int(entry["anno"])
            key = entry["frame_key"]
            graph_path = graph_root / entry["graph_path"]
            if not graph_path.is_file() or sha256_file(graph_path) != entry["graph_sha256"]:
                raise RuntimeError(f"{method}/{key}: graph missing/corrupt")
            with np.load(graph_path, allow_pickle=False) as z:
                g = np.asarray(z["grasps"], dtype=np.float32)
                pids = np.asarray(z["physical_id"]).astype(str)
                edges = np.asarray(z["edges"], dtype=np.int64)
                component_id = np.asarray(z["component_id"], dtype=np.int64)
                native_keep = np.asarray(z["nms_keep"], dtype=np.int64)
                suppressor = np.asarray(z["suppressor"], dtype=np.int64)
                if not np.array_equal(physical_ids(g), pids):
                    raise RuntimeError(f"{method}/{key}: physical ID mismatch")

            label_record = label_records.get(key)
            if label_record is None:
                raise RuntimeError(f"No exact-action label cache for {key}")
            labels = _load_label(labels_root, label_record, gm["manifest_sha256"])
            label_map = {pid: i for i, pid in enumerate(labels["physical_id"].astype(str))}
            try:
                label_index = np.asarray([label_map[x] for x in pids], dtype=np.int64)
            except KeyError as exc:
                raise RuntimeError(f"{method}/{key}: candidate lacks exact label {exc}") from exc
            assigned = np.asarray(labels["assigned_obj"], dtype=np.int64)[label_index]
            bad = np.asarray(labels["collision_or_empty"], dtype=bool)[label_index]
            friction = np.asarray(labels["friction"], dtype=np.float64)[label_index]
            bits = np.asarray(labels["success_bits"], dtype=np.uint8)[label_index]
            q_count = bits.sum(axis=1).astype(np.int64)
            if not np.array_equal(q_count, np.asarray(labels["quality_count"])[label_index]):
                raise RuntimeError(f"{method}/{key}: quality bits/count mismatch")

            aps, metrics = _frame_ap(
                g, assigned, bad, friction, bits, edges, component_id,
                native_keep, suppressor, a.max_exact_component_nodes,
            )
            if official is not None:
                si = scene - int(gm["protocol"]["scenes"][0])
                if not (0 <= si < 30 and 0 <= anno < 256):
                    raise ValueError(f"Frame outside official AP array: {key}")
                err = float(np.max(np.abs(aps["native"] - official[si, anno])))
                parity_errors.append(err)
                if err > a.ap_tolerance and not a.skip_official_parity:
                    raise RuntimeError(
                        f"{method}/{key}: native AP replay mismatch {err:.8g} > {a.ap_tolerance}; "
                        "do not interpret oracle AP until evaluator parity is fixed"
                    )

            for variant in VARIANTS:
                frame_ap[variant].append(aps[variant])
            frame_native = float(aps["native"].mean() * 100.0)
            row = {
                "method": method, "scene": scene, "anno": anno,
                "num_candidates": len(g), "num_edges": len(edges),
                "num_components": int(component_id.max() + 1) if len(component_id) else 0,
                "nms_survivors": len(native_keep),
                "mean_candidate_q": float(q_count.mean() / 6.0) if len(q_count) else 0.0,
                "native_survivor_q": float(q_count[native_keep].mean() / 6.0) if len(native_keep) else 0.0,
                "quality_nms_survivor_q": float(q_count[greedy_nms(q_priority(q_count, g[:, 0]), edges)[0]].mean() / 6.0) if len(g) else 0.0,
                "suppressed_q_better": metrics["suppressed_candidates_with_higher_q_than_suppressor"],
                "suppressed_candidates": metrics["suppressed_candidates"],
                "component_exact_node_coverage": metrics["component_oracle"]["exact_node_coverage"],
                "native_ap_percent": frame_native,
            }
            for variant in VARIANTS:
                row[f"{variant}_ap_percent"] = float(aps[variant].mean() * 100.0)
            per_frame_rows.append(row)
            frame_rows.append(row)

            graph_totals["nodes"] += len(g)
            graph_totals["edges"] += len(edges)
            graph_totals["components"] += int(component_id.max() + 1) if len(component_id) else 0
            graph_totals["native_survivors"] += len(native_keep)
            selection_totals["suppressed"] += metrics["suppressed_candidates"]
            selection_totals["suppressed_q_better"] += metrics["suppressed_candidates_with_higher_q_than_suppressor"]
            selection_totals["component_exact_nodes"] += metrics["component_oracle"]["exact_nodes"]
            selection_totals["component_total_nodes"] += metrics["component_oracle"]["total_nodes"]
            selection_totals["component_exact_components"] += metrics["component_oracle"]["exact_components"]
            selection_totals["component_fallback_components"] += metrics["component_oracle"]["fallback_components"]

            per_scene_acc.setdefault(scene, {v: [] for v in VARIANTS})
            for variant in VARIANTS:
                per_scene_acc[scene][variant].append(float(aps[variant].mean()))

            if frame_number % 256 == 0 or frame_number == len(entries):
                print(f"[analyze:{method}] {frame_number}/{len(entries)} {key}", flush=True)

        for variant in VARIANTS:
            frame_ap[variant] = np.stack(frame_ap[variant], axis=0)
        scene_ap = {
            scene: {
                variant: float(np.mean(per_scene_acc[scene][variant]))
                for variant in VARIANTS
            } for scene in per_scene_acc
        }
        mean_ap = {variant: _mean_ap(frame_ap[variant]) for variant in VARIANTS}
        boot = {
            variant: _bootstrap_delta(
                {scene: {"native": scene_ap[scene]["native"], variant: scene_ap[scene][variant]}
                 for scene in scene_ap},
                variant, a.bootstrap, a.seed,
            ) for variant in VARIANTS if variant != "native"
        }
        parity = {
            "passed": bool(official is not None and max(parity_errors, default=0.0) <= a.ap_tolerance),
            "max_abs": max(parity_errors, default=None),
            "mean_abs": float(np.mean(parity_errors)) if parity_errors else None,
            "tolerance": float(a.ap_tolerance),
            "official_ap_path": str(official_path) if official_path else None,
        }
        method_summary = {
            "num_frames": len(entries),
            "graph_totals": graph_totals,
            "selection_totals": {
                **selection_totals,
                "fraction_suppressed_q_better": (
                    selection_totals["suppressed_q_better"] / selection_totals["suppressed"]
                    if selection_totals["suppressed"] else 0.0
                ),
                "component_exact_node_coverage": (
                    selection_totals["component_exact_nodes"] / selection_totals["component_total_nodes"]
                    if selection_totals["component_total_nodes"] else 1.0
                ),
                "component_oracle_globally_exact": selection_totals["component_fallback_components"] == 0,
            },
            "mean_ap_percent": mean_ap,
            "scene_bootstrap": boot,
            "official_ap_parity": parity,
            "ap_by_mu_percent": {
                variant: frame_ap[variant].mean(axis=(0, 1)).tolist()
                for variant in VARIANTS
            },
        }
        summary["methods"][method] = method_summary
        for scene in sorted(scene_ap):
            scene_rows.append({
                "method": method, "scene": scene,
                **{f"{v}_ap_percent": scene_ap[scene][v] * 100.0 for v in VARIANTS},
            })
        method_out = out_root / method
        method_out.mkdir(parents=True, exist_ok=True)
        write_npz(method_out / "frame_ap.npz", **{
            v: frame_ap[v].astype(np.float32) for v in VARIANTS
        })
        _write = per_frame_rows
        _write_csv = _write
        # Keep per-method CSVs as well as the combined file.
        keys = list(_write[0]) if _write else []
        with (method_out / "per_frame.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if keys:
                writer.writeheader()
                writer.writerows(_write)

    # A formal run is not complete if any method's native AP replay differs.
    if not a.skip_official_parity:
        failed = [m for m, r in summary["methods"].items() if not r["official_ap_parity"]["passed"]]
        if failed:
            raise RuntimeError(f"Official AP parity failed for methods: {failed}")

    write_json(out_root / "summary.json", summary)
    _write_csv(out_root / "per_frame.csv", frame_rows)
    _write_csv(out_root / "per_scene.csv", scene_rows)
    (out_root / "RESULTS_CN.md").write_text(_render_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "split": gm["split"],
        "methods": {
            m: {
                "native_ap": r["mean_ap_percent"]["native"],
                "quality_nms_native_rank": r["mean_ap_percent"]["quality_nms_native_rank"],
                "component_set_native_rank": r["mean_ap_percent"]["component_set_native_rank"],
                "per_mu_success_oracle": r["mean_ap_percent"]["per_mu_success_oracle"],
                "parity": r["official_ap_parity"],
            } for m, r in summary["methods"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    import csv
    main()
