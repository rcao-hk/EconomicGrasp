"""Exact CAD/Dex-Net labels for the union of candidate actions across methods."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .core import (
    SCHEMA, digest, physical_ids, quality_bits, sha256_file, write_json, write_npz,
)


def load_verified_manifest(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    actual = obj.pop("manifest_sha256", None)
    if actual is None or digest(obj) != actual:
        raise ValueError("Graph manifest fingerprint mismatch")
    obj["manifest_sha256"] = actual
    if obj.get("schema") != SCHEMA:
        raise ValueError("Wrong graph manifest schema")
    return obj


def _unique_actions(graph_root: Path, entries_by_method: dict, frame_key: str):
    by_id = {}
    graph_shas = {}
    for method, entry in entries_by_method.items():
        graph_path = graph_root / entry["graph_path"]
        if not graph_path.is_file() or sha256_file(graph_path) != entry["graph_sha256"]:
            raise RuntimeError(f"Missing/corrupt graph: {graph_path}")
        graph_shas[method] = entry["graph_sha256"]
        with np.load(graph_path, allow_pickle=False) as z:
            grasps = np.asarray(z["grasps"], dtype=np.float32)
            ids = np.asarray(z["physical_id"]).astype(str)
            if len(ids) != len(grasps):
                raise ValueError(f"{frame_key}/{method}: physical ID count mismatch")
            if not np.array_equal(physical_ids(grasps), ids):
                raise ValueError(f"{frame_key}/{method}: physical ID recomputation mismatch")
            for row, pid in zip(grasps, ids):
                if pid in by_id:
                    if not np.array_equal(by_id[pid], row[1:16]):
                        raise RuntimeError(f"Physical ID collision/inconsistent geometry: {frame_key}/{pid}")
                else:
                    by_id[pid] = row[1:16].copy()
    ids = np.asarray(sorted(by_id), dtype="<U64")
    grasps = np.zeros((len(ids), 17), dtype=np.float32)
    for i, pid in enumerate(ids):
        grasps[i, 1:16] = by_id[pid]
        grasps[i, 0] = 0.0  # evaluator ignores confidence; source scores remain in graph files
        grasps[i, 16] = -1.0
    return ids, grasps, graph_shas


def _label_scene(task: dict) -> dict:
    """One process handles one scene, reusing its CAD/Dex-Net scene model cache."""
    from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator

    graph_root = Path(task["graph_root"])
    output_root = Path(task["output_root"])
    scene = int(task["scene"])
    evaluator = ExactGraspNetActionEvaluator(
        task["dataset_root"], task["camera"], split=task["split"],
        fc_mode="official", strict=True,
    )
    results = []
    for frame in task["frames"]:
        anno = int(frame["anno"])
        key = frame["frame_key"]
        out = output_root / "labels" / f"scene_{scene:04d}" / f"{anno:04d}.npz"
        meta = out.with_suffix(".json")
        ids, grasps, graph_shas = _unique_actions(
            graph_root, frame["entries_by_method"], key
        )
        contract = {
            "schema": SCHEMA,
            "frame_key": key,
            "scene": scene,
            "anno": anno,
            "split": task["split"],
            "camera": task["camera"],
            "graph_manifest_sha256": task["graph_manifest_sha256"],
            "graph_sha256_by_method": graph_shas,
            "evaluator": "ExactGraspNetActionEvaluator",
            "evaluator_source_sha256": task["evaluator_source_sha256"],
            "fc_mode": "official",
            "width_clip_m": float(task["max_width"]),
            "num_unique_actions": int(len(grasps)),
        }
        contract_sha = digest(contract)
        if out.exists() or meta.exists():
            if not (out.exists() and meta.exists()):
                raise RuntimeError(f"Incomplete label cache: {out}")
            old = json.loads(meta.read_text(encoding="utf-8"))
            if old.get("contract_sha256") != contract_sha:
                raise RuntimeError(f"Label cache contract mismatch: {out}")
            if sha256_file(out) != old.get("labels_sha256"):
                raise RuntimeError(f"Label cache hash mismatch: {out}")
            results.append(old)
            continue

        ev = evaluator.evaluate(scene, anno, grasps)
        bits = quality_bits(ev.friction, ev.collision_or_empty)
        quality_count = bits.sum(axis=1).astype(np.uint8)
        write_npz(
            out,
            physical_id=ids,
            grasps=grasps.astype(np.float32),
            assigned_obj=np.asarray(ev.assigned_obj, dtype=np.int32),
            collision_or_empty=np.asarray(ev.collision_or_empty, dtype=np.uint8),
            pure_collision=np.asarray(ev.pure_collision, dtype=np.uint8),
            empty=np.asarray(ev.empty, dtype=np.uint8),
            friction=np.asarray(ev.friction, dtype=np.float32),
            success_bits=bits,
            quality_count=quality_count,
        )
        record = {
            "contract": contract,
            "contract_sha256": contract_sha,
            "labels_path": str(out.relative_to(output_root)),
            "labels_sha256": sha256_file(out),
            "stats": ev.stats,
        }
        write_json(meta, record)
        results.append(record)
        print(
            f"[labels] scene={scene} anno={anno} unique_actions={len(grasps)} "
            f"fc={ev.stats.get('fc_candidates', 0)}",
            flush=True,
        )
    return {"scene": scene, "frames": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-root", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--camera", default="realsense", choices=("realsense", "kinect"))
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=1,
                   help="Process workers by scene; each worker loads official CAD/Dex-Net models.")
    p.add_argument("--allow-partial", action="store_true",
                   help="Smoke only; formal oracle analysis requires a complete 100% graph manifest.")
    a = p.parse_args()

    graph_root = Path(a.graph_root)
    graph_manifest_path = graph_root / "graph_manifest.json"
    manifest = load_verified_manifest(graph_manifest_path)
    if not manifest.get("complete") and not a.allow_partial:
        raise RuntimeError("Graph manifest is partial; formal quality labels require all 256 frames/scene")
    if manifest["camera"] != a.camera:
        raise ValueError("Camera mismatch between graph manifest and label command")
    if a.workers < 1:
        raise ValueError("--workers must be positive")
    split = manifest["split"]
    methods = manifest["methods"]
    frame_maps = {}
    for method, item in methods.items():
        frame_map = {}
        for entry in item["entries"]:
            key = entry["frame_key"]
            if key in frame_map:
                raise ValueError(f"Duplicate frame in {method}: {key}")
            frame_map[key] = entry
        frame_maps[method] = frame_map
    common_keys = set.intersection(*(set(x) for x in frame_maps.values()))
    if any(set(x) != common_keys for x in frame_maps.values()):
        raise ValueError("All methods must have identical frame manifests")
    frames_by_scene = {}
    for key in sorted(common_keys):
        # scene_0100_0000
        left, anno_s = key.rsplit("_", 1)
        scene = int(left.split("_")[-1])
        anno = int(anno_s)
        frames_by_scene.setdefault(scene, []).append({
            "frame_key": key,
            "anno": anno,
            "entries_by_method": {m: frame_maps[m][key] for m in methods},
        })

    out_root = Path(a.output)
    out_root.mkdir(parents=True, exist_ok=True)
    evaluator_path = Path(__file__).resolve().parents[2] / "exact_action_graspnet_evaluator.py"
    if not evaluator_path.is_file():
        raise FileNotFoundError(f"Expected repository evaluator at {evaluator_path}")
    tasks = []
    for scene, frames in sorted(frames_by_scene.items()):
        tasks.append({
            "graph_root": str(graph_root.resolve()),
            "output_root": str(out_root.resolve()),
            "dataset_root": str(Path(a.dataset_root).resolve()),
            "camera": a.camera,
            "split": split,
            "scene": scene,
            "frames": frames,
            "graph_manifest_sha256": manifest["manifest_sha256"],
            "evaluator_source_sha256": sha256_file(evaluator_path),
            "max_width": manifest["protocol"]["max_width_m"],
        })

    scene_records = []
    if a.workers == 1:
        for task in tasks:
            scene_records.append(_label_scene(task))
    else:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(_label_scene, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                scene_records.append(future.result())
                print(f"[labels] completed scenes {i}/{len(tasks)}", flush=True)
    scene_records.sort(key=lambda x: x["scene"])
    all_records = [r for scene in scene_records for r in scene["frames"]]
    expected = len(common_keys)
    if len(all_records) != expected:
        raise RuntimeError(f"Label count mismatch: {len(all_records)} != {expected}")
    label_manifest = {
        "schema": SCHEMA,
        "status": "complete" if manifest.get("complete") else "partial_smoke",
        "complete": bool(manifest.get("complete")),
        "split": split,
        "camera": a.camera,
        "graph_manifest_path": str(graph_manifest_path.resolve()),
        "graph_manifest_sha256": manifest["manifest_sha256"],
        "dataset_root": str(Path(a.dataset_root).resolve()),
        "fc_mode": "official",
        "workers": int(a.workers),
        "num_methods": len(methods),
        "methods": list(methods),
        "num_frames": len(all_records),
        "num_unique_actions": int(sum(r["contract"]["num_unique_actions"] for r in all_records)),
        "scene_records": scene_records,
    }
    label_manifest["manifest_sha256"] = digest(label_manifest)
    write_json(out_root / "label_manifest.json", label_manifest)
    print(json.dumps({
        "status": label_manifest["status"],
        "frames": label_manifest["num_frames"],
        "unique_actions": label_manifest["num_unique_actions"],
        "workers": a.workers,
    }, indent=2))


if __name__ == "__main__":
    main()
