"""Build complete per-method GraspNet NMS conflict graphs from pre-NMS dumps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import (
    SCHEMA, build_conflict_edges, connected_components, digest, frame_key,
    graph_stats, greedy_nms, load_grasps, physical_ids, sha256_file,
    verify_official_nms, write_json, write_npz,
)

SPLITS = {
    "test_seen": range(100, 130),
    "test_similar": range(130, 160),
    "test_novel": range(160, 190),
}


def load_verified_manifest(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    actual = obj.pop("manifest_sha256", None)
    if actual is None or digest(obj) != actual:
        raise ValueError("Graph manifest fingerprint mismatch")
    obj["manifest_sha256"] = actual
    if obj.get("schema") != SCHEMA:
        raise ValueError("Wrong graph manifest schema")
    return obj


def parse_named(values, field):
    out = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"{field} must use NAME=PATH syntax, got {item!r}")
        name, value = item.split("=", 1)
        if not name or not value or name in out:
            raise ValueError(f"Invalid/duplicate {field}: {item!r}")
        out[name] = value
    return out


def _frame_paths(root: Path, scene: int, camera: str, anno: int):
    return root / f"scene_{scene:04d}" / camera / f"{anno:04d}.npy"


def _build_one(
    method: str, dump_root: Path, out_root: Path, scene: int, anno: int,
    camera: str, max_width: float, trans_thresh: float, rot_rad: float,
    input_stage: str, verify_nms: bool, resume: bool, protocol_sha: str,
):
    source = _frame_paths(dump_root, scene, camera, anno)
    if not source.is_file():
        raise FileNotFoundError(f"Missing candidate dump: {source}")
    source_sha = sha256_file(source)
    out = out_root / "graphs" / method / f"scene_{scene:04d}" / f"{anno:04d}.npz"
    meta = out.with_suffix(".json")
    if out.exists() or meta.exists():
        if not resume or not (out.exists() and meta.exists()):
            raise FileExistsError(f"Existing/incomplete graph cache: {out}; use --resume only after inspection")
        old = json.loads(meta.read_text(encoding="utf-8"))
        if old.get("source_sha256") != source_sha or old.get("protocol_sha256") != protocol_sha:
            raise RuntimeError(f"Resume contract mismatch for {source}")
        if sha256_file(out) != old.get("graph_sha256"):
            raise RuntimeError(f"Existing graph cache hash mismatch: {out}")
        return old

    grasps, width_stats = load_grasps(source, max_width=max_width)
    edges = build_conflict_edges(grasps, trans_thresh, rot_rad)
    component_id = connected_components(len(grasps), edges)
    kept, suppressor, order = greedy_nms(grasps[:, 0], edges)
    parity = None
    if verify_nms:
        parity = verify_official_nms(grasps, kept, trans_thresh, rot_rad)

    stats = graph_stats(len(grasps), edges, component_id, kept)
    stats.update(width_stats)
    write_npz(
        out,
        grasps=grasps.astype(np.float32),
        physical_id=physical_ids(grasps),
        source_row=np.arange(len(grasps), dtype=np.int64),
        edges=edges.astype(np.int64),
        component_id=component_id.astype(np.int64),
        nms_keep=kept.astype(np.int64),
        suppressor=suppressor.astype(np.int64),
        nms_order=order.astype(np.int64),
    )
    record = {
        "method": method,
        "scene": int(scene),
        "anno": int(anno),
        "frame_key": frame_key(scene, anno),
        "source_path": str(source.resolve()),
        "source_sha256": source_sha,
        "graph_path": str(out.relative_to(out_root)),
        "graph_sha256": sha256_file(out),
        "protocol_sha256": protocol_sha,
        "input_stage": input_stage,
        "official_nms_parity_max_abs": parity,
        "stats": stats,
    }
    write_json(meta, record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", required=True, choices=tuple(SPLITS))
    p.add_argument("--camera", default="realsense", choices=("realsense", "kinect"))
    p.add_argument("--method", action="append", required=True,
                   help="Repeat NAME=/path/to/pre-NMS/dump-root; root contains scene_XXXX/CAMERA/NNNN.npy")
    p.add_argument("--official-ap", action="append", default=[],
                   help="Optional repeat NAME=/path/to/ap_split_camera.npy")
    p.add_argument("--output", required=True)
    p.add_argument("--input-stage", required=True, choices=("sensor_filtered_pre_nms",),
                   help="Explicit assertion: inputs must be before official GraspNet NMS")
    p.add_argument("--max-width", type=float, default=0.1)
    p.add_argument("--translation-thresh", type=float, default=0.03)
    p.add_argument("--rotation-thresh-deg", type=float, default=30.0)
    p.add_argument("--scenes", default="", help="Smoke/debug subset; comma-separated scene IDs")
    p.add_argument("--anno-start", type=int, default=0)
    p.add_argument("--anno-stop", type=int, default=256)
    p.add_argument("--anno-step", type=int, default=1)
    p.add_argument("--limit-frames", type=int, default=0, help="Debug only; marks manifest incomplete")
    p.add_argument("--no-official-nms-check", action="store_true",
                   help="Smoke only. Formal runs must verify installed GraspGroup.nms parity.")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    methods = parse_named(a.method, "--method")
    official_ap = parse_named(a.official_ap, "--official-ap")
    if set(official_ap) - set(methods):
        raise ValueError("An --official-ap method has no matching --method")
    scenes = list(SPLITS[a.split])
    if a.scenes:
        requested = [int(x) for x in a.scenes.split(",") if x]
        if not requested or any(x not in SPLITS[a.split] for x in requested):
            raise ValueError("--scenes must be a nonempty subset of the requested split")
        scenes = requested
    if not (0 <= a.anno_start < a.anno_stop <= 256) or a.anno_step < 1:
        raise ValueError("Invalid annotation range")
    annos = list(range(a.anno_start, a.anno_stop, a.anno_step))
    frames = [(s, n) for s in scenes for n in annos]
    if a.limit_frames:
        if a.limit_frames < 1:
            raise ValueError("--limit-frames must be positive")
        frames = frames[:a.limit_frames]
    if not frames:
        raise ValueError("No frames selected")
    expected_full = len(SPLITS[a.split]) * 256
    complete = (len(frames) == expected_full and scenes == list(SPLITS[a.split])
                and annos == list(range(256)) and not a.limit_frames)

    out_root = Path(a.output)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "graph_manifest.json"
    if manifest_path.exists() and not a.resume:
        raise FileExistsError(f"{manifest_path} exists; use a new output directory or explicit --resume")

    rot_rad = np.deg2rad(a.rotation_thresh_deg)
    protocol = {
        "schema": SCHEMA,
        "split": a.split,
        "camera": a.camera,
        "input_stage": a.input_stage,
        "max_width_m": float(a.max_width),
        "translation_thresh_m": float(a.translation_thresh),
        "rotation_thresh_deg": float(a.rotation_thresh_deg),
        "scenes": scenes,
        "annos": annos,
        "expected_full_frames": expected_full,
        "selected_frames": len(frames),
        "methods": {k: str(Path(v).resolve()) for k, v in methods.items()},
        "official_ap": {k: str(Path(v).resolve()) for k, v in official_ap.items()},
        "official_nms_check": not a.no_official_nms_check,
    }
    protocol_sha = digest(protocol)
    old_manifest = None
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("protocol_sha256") != protocol_sha:
            raise RuntimeError("Resume graph manifest protocol differs; use a new output directory")

    records = {name: [] for name in methods}
    if old_manifest:
        for name, values in old_manifest.get("methods", {}).items():
            if name in records:
                records[name] = list(values.get("entries", []))
    for method, root_s in methods.items():
        root = Path(root_s)
        already = {(int(x["scene"]), int(x["anno"])): x for x in records[method]}
        for index, (scene, anno) in enumerate(frames, 1):
            if (scene, anno) in already:
                continue
            record = _build_one(
                method, root, out_root, scene, anno, a.camera, a.max_width,
                a.translation_thresh, rot_rad, a.input_stage,
                not a.no_official_nms_check, a.resume, protocol_sha,
            )
            records[method].append(record)
            if index % 32 == 0 or index == len(frames):
                print(f"[graph:{method}] {index}/{len(frames)} "
                      f"scene={scene} anno={anno} nodes={record['stats']['num_nodes']} "
                      f"edges={record['stats']['num_edges']}", flush=True)

    for method in records:
        records[method].sort(key=lambda x: (x["scene"], x["anno"]))
        if len(records[method]) != len(frames):
            raise RuntimeError(f"{method}: graph count {len(records[method])} != requested {len(frames)}")
        for rec in records[method]:
            graph_file = out_root / rec["graph_path"]
            if not graph_file.is_file() or sha256_file(graph_file) != rec["graph_sha256"]:
                raise RuntimeError(f"Graph artifact missing/corrupt: {graph_file}")

    manifest = {
        "schema": SCHEMA,
        "status": "complete" if complete else "partial_smoke",
        "complete": bool(complete),
        "split": a.split,
        "camera": a.camera,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "methods": {
            name: {
                "dump_root": str(Path(methods[name]).resolve()),
                "official_ap_path": str(Path(official_ap[name]).resolve()) if name in official_ap else None,
                "entries": records[name],
                "num_frames": len(records[name]),
                "num_nodes": int(sum(x["stats"]["num_nodes"] for x in records[name])),
                "num_edges": int(sum(x["stats"]["num_edges"] for x in records[name])),
                "nms_parity_max_abs": max(
                    [x["official_nms_parity_max_abs"] or 0.0 for x in records[name]], default=0.0
                ),
            } for name in methods
        },
    }
    manifest["manifest_sha256"] = digest(manifest)
    write_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"], "split": a.split,
        "frames_per_method": len(frames), "methods": list(methods),
        "nodes": {k: v["num_nodes"] for k, v in manifest["methods"].items()},
        "edges": {k: v["num_edges"] for k, v in manifest["methods"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
