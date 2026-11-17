"""Frozen GC6D RGB -> exact seven-action bank -> P6-2A export.

Stages: plan, parity-graspnet, build, audit, evaluate. Previous experiments,
weights, dataset files and quality definitions remain read-only. No training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .p62_gc6d_contract import (SCHEMA, GENERATOR, OFFSETS, CAMERAS, sha256, digest,
    signed, load_signed, atomic_json, atomic_npz, image_id, seed_for, choose_families,
    physical_ids, validate_bank, quality_counts, validate_frame, check_lineage, verify_artifact, cad_paths)


def artifact(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha256(p))


def source_code():
    root = Path(__file__).resolve().parents[1]
    paths = ["shared_geometry/p62_gc6d.py", "shared_geometry/p62_gc6d_contract.py",
             "shared_geometry/p62_gc6d_runtime.py", "shared_geometry/geometry.py",
             "shared_geometry/runtime.py", "shared_geometry/p4_depth_baseline.py",
             "interactive_grasp/native.py", "interactive_grasp/capture.py",
             "exact_action_graspnet_evaluator.py"]
    # Source model/dataset utility imports are pinned as well as the new wrapper.
    for folder in ("models", "utils", "rayrope_gvar", "libs"):
        if (root / folder).is_dir():
            paths += [str(p.relative_to(root)) for p in (root / folder).rglob("*.py")]
    return {str((root / p).resolve()): sha256(root / p) for p in sorted(set(paths)) if (root / p).is_file()}


def bind_inputs(config, p1_index, depth_index, depth_weight, pred_weight, corr_weight):
    import torch
    cfg = json.loads(Path(config).read_text())
    assets = {"base": artifact(cfg["base_checkpoint"]), "selector": artifact(cfg["selector_checkpoint"]),
              "depth_corrector": artifact(depth_weight), "predicted_reader": artifact(pred_weight),
              "corrected_reader": artifact(corr_weight)}
    for name in ("base", "selector"):
        if assets[name]["sha256"] != cfg[name+"_sha256"]:
            raise ValueError("Bound native source weights changed")
    pi, di = load_signed(p1_index), load_signed(depth_index)
    if pi["partition"] != "validation" or not pi["complete"] or di["partition"] != "validation" or not di["complete"]:
        raise ValueError("Need the completed P1/P4 reference validation indices")
    for key in ("base_sha256", "selector_sha256", "config_sha256"):
        expected = sha256(config) if key == "config_sha256" else cfg[key]
        if pi["contract"]["source"].get(key) != expected:
            raise ValueError(f"Provider is not the P1/P5 frozen source: {key}")
    readers = []
    selections = []
    for arm, path in (("predicted", pred_weight), ("corrected", corr_weight)):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        p = ck["protocol"]
        selpath = Path(path).parent / "selection.json"; sel = load_signed(selpath)
        if (ck.get("schema") != "shared-visible-geometry-p5-reader-v1" or p.get("arm") != arm
            or p.get("smoke", True) or p["depth_weight_sha256"] != assets["depth_corrector"]["sha256"]
            or p["p1_fingerprint"] != pi["fingerprint"] or p["source"] != pi["contract"]["source"]
            or p["teacher_config"] != pi["contract"]["target_config"] or not sel["complete"]
            or sel["weight_sha256"] != sha256(path)):
            raise ValueError(f"Unselected/mismatched formal P5 {arm} reader")
        readers.append(p); selections.append(artifact(selpath))
    for key in ("train_fingerprint", "val_fingerprint", "cfg", "seed", "epochs", "batch_size", "accumulate", "lr", "weight_decay"):
        if readers[0][key] != readers[1][key]:
            raise ValueError(f"Reader controls differ: {key}")
    depth_ck = torch.load(depth_weight, map_location="cpu", weights_only=False)
    dp = depth_ck["protocol"]
    depth_sel = Path(depth_weight).parent / "selection.json"; ds = load_signed(depth_sel)
    if (dp.get("smoke", True) or dp["val_fingerprint"] != di["fingerprint"]
        or dp["teacher_config"] != pi["contract"]["target_config"]
        or dp["train_source"] != pi["contract"]["source"] or not ds["complete"]
        or ds["best_sha256"] != sha256(depth_weight) or ds["selected_epoch"] != depth_ck["epoch"]
        or not ds["no_action_quality_used_for_selection"]):
        raise ValueError("P4 corrector is not the completed depth-only selected reference")
    tau, band = float(pi["contract"]["tau"]), float(pi["contract"]["band"])
    if tuple(pi["contract"]["offsets_m"]) != OFFSETS or not 0 < band < tau or readers[0]["cfg"]["truncation_m"] != tau:
        raise ValueError("Historical geometry contract differs")
    e = di["entries"][0]
    verify_artifact(dict(path=e["source_path"], sha256=e["source_sha256"]))
    with np.load(e["source_path"], allow_pickle=False) as z:
        feature_dtype = z["feature_map"].dtype.name
        channels = z["feature_map"].shape[0]
    if feature_dtype not in ("float16", "float32") or channels != dp["architecture"]["feature_channels"]:
        raise ValueError("P4 feature-storage/channel contract unavailable")
    return dict(assets=assets, config=artifact(config), p1_index=artifact(p1_index),
                depth_index=artifact(depth_index), selections=selections+[artifact(depth_sel)],
                feature_reference=dict(path=e["source_path"], sha256=e["source_sha256"]),
                feature_dtype=feature_dtype, channels=channels, tau=tau, band=band,
                p1_fingerprint=pi["fingerprint"], depth_index_fingerprint=di["fingerprint"],
                source=pi["contract"]["source"], reader_protocols=readers)


def plan(a):
    if a.camera not in CAMERAS or a.families < 1 or a.families > 1024 or a.batch_size < 1:
        raise ValueError("Bad camera/family/batch setting")
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    binding = bind_inputs(a.config, a.p1_index, a.depth_index, a.depth_weight,
                          a.predicted_weight, a.corrected_weight)
    lineage = json.loads(Path(a.lineage).read_text())
    scene_list = json.loads(Path(a.scenes_json).read_text())
    if not isinstance(scene_list, list) or not scene_list or not all(isinstance(v, (int, str)) and str(v).isdigit() for v in scene_list):
        raise ValueError("scenes-json must be a nonempty JSON list of GC6D scene IDs")
    scenes = sorted(map(int, scene_list))
    if len(set(scenes)) != len(scenes) or a.scene_count < 1 or a.scene_count > len(scenes):
        raise ValueError("Duplicate scenes or invalid scene count")
    # Require the supplied pool itself to be unseen; do not silently remove exposed scenes.
    check_lineage(lineage, binding["assets"], scenes)
    scenes = sorted(np.random.default_rng(a.seed).choice(scenes, a.scene_count, replace=False).tolist())
    annos = [int(v) for v in a.annos.split(",")]
    if len(set(annos)) != len(annos) or not annos:
        raise ValueError("Repeated/empty annos")
    for v in annos:
        image_id(v, a.camera)
    rows = []
    root = Path(a.dataset_root).resolve()
    for scene in scenes:
        folder = root / "scenes" / f"{scene:06d}"
        cameras = json.loads((folder / "scene_camera.json").read_text())
        gt = json.loads((folder / "scene_gt.json").read_text())
        for anno in annos:
            img = image_id(anno, a.camera)
            if str(img) not in cameras or str(img) not in gt:
                raise ValueError(f"Missing calibrated/annotated frame: {scene}/{img}")
            files = [folder/"rgb"/f"{img:06d}.png", folder/"scene_camera.json", folder/"scene_gt.json"]
            for obj in gt[str(img)]:
                oid = int(obj["obj_id"])
                files.extend(cad_paths(root, oid))
            rows.append(dict(scene=scene, anno=anno, img_id=img, domain="original", split="external",
                             partition="independent_holdout", input_files=[artifact(p) for p in sorted(set(files))]))
    record = dict(schema="p62-gc6d-plan-v1", binding=binding, lineage=lineage,
                  lineage_file=artifact(a.lineage), scenes_file=artifact(a.scenes_json),
                  dataset_root=str(root), camera=a.camera, rows=rows, seed=a.seed,
                  families=a.families, offsets_m=list(OFFSETS), batch_size=a.batch_size,
                  crop_box=a.crop_box, code_hashes=source_code(),
                  scope="external_scene_holdout_not_unseen_object_claim",
                  no_sensor_depth_or_mask_input=True, formal_not_smoke=True)
    atomic_json(a.output, signed(record))
    return record


def verify_plan(p):
    if p.get("schema") != "p62-gc6d-plan-v1" or not p.get("formal_not_smoke") or p["offsets_m"] != list(OFFSETS):
        raise ValueError("Unexpected frozen plan")
    b = p["binding"]
    for x in list(b["assets"].values()) + b["selections"] + [b[k] for k in ("config", "p1_index", "depth_index", "feature_reference")]:
        verify_artifact(x)
    for path, checksum in p["code_hashes"].items():
        if sha256(path) != checksum:
            raise ValueError(f"Code changed since plan: {path}")
    check_lineage(p["lineage"], b["assets"], [r["scene"] for r in p["rows"]])


def check_parity(report, p):
    if not report:
        raise ValueError("--parity-report is required for a formal export")
    r = load_signed(report)
    if (r.get("schema") != "p62-graspnet-parity-v1" or r.get("mismatches") != 0
        or r.get("n_positive", 0) < 1 or r.get("plan_fingerprint") != p["fingerprint"]
        or r.get("n_actions", 0) < 7):
        raise ValueError("Formal export requires non-vacuous matched GraspNet exact-label parity")
    verify_artifact(r["original_index"])
    for entry in r["reference_frames"]:
        verify_artifact(entry)
    return r


def parity_graspnet(a):
    """Replay original signed-bank labels and cross-check GC6D vs GN physics kernels."""
    from exact_action_graspnet_evaluator import ExactGraspNetActionEvaluator
    from .p62_gc6d_runtime import ExactGC6D
    p = load_signed(a.plan); verify_plan(p)
    index = load_signed(a.source_index)
    if index["contract"]["source"] != p["binding"]["source"] or index["partition"] != "validation":
        raise ValueError("Use original frozen-source development bank, not official test")
    if a.frames < 1 or a.frames > len(index["entries"]) or Path(a.output).exists():
        raise ValueError("Bad parity budget/output exists")
    gn = ExactGraspNetActionEvaluator(a.graspnet_root, "realsense", split="train", fc_mode="official", strict=True)
    gc = ExactGC6D(p["dataset_root"], p["camera"], a.chunk, 3)
    n, positive, sources = 0, 0, []
    # Round-robin domains; label truth is not inspected when picking reference frames.
    entries = index["entries"]; domains = sorted({e["row"]["domain"] for e in entries})
    queues = [[e for e in entries if e["row"]["domain"] == d] for d in domains]
    chosen = [batch[i] for i in range(max(map(len, queues))) for batch in queues if i < len(batch)][:a.frames]
    for e in chosen:
        verify_artifact(e)
        row = e["row"]
        if not 0 <= row["scene"] < 100:
            raise ValueError("Parity must never read official test")
        with np.load(e["path"], allow_pickle=False) as z:
            bank, expected = z["bank"].copy(), z["quality_count"].copy()
        validate_bank(bank); g = bank.reshape(-1, 17)
        x = gn.evaluate(row["scene"], row["anno"], g)
        count = quality_counts(x.friction, np.asarray(x.collision_or_empty, bool))
        models, dex = gn._scene_models(row["scene"])
        _, poses, camera_pose, align = gn.eval.get_model_poses(row["scene"], row["anno"])
        table = gc.eu.transform_points(gn.table, np.linalg.inv(align @ camera_pose))
        y = gc.evaluate_arrays(g, models, dex, poses, table)
        if (not np.array_equal(count.reshape(expected.shape), expected) or
            not np.array_equal(count, y["quality_count"]) or
            not np.array_equal(x.collision_or_empty, y["collision_or_empty"]) or
            not np.allclose(x.friction, y["friction"], atol=1e-6, rtol=0)):
            raise RuntimeError(f"Historical/GN/GC exact-label parity failed: {row}")
        n += len(g); positive += int((count > 0).sum()); sources.append(artifact(e["path"]))
    if positive == 0:
        raise RuntimeError("All-zero parity is inconclusive; preregister more reference frames")
    r = dict(schema="p62-graspnet-parity-v1", plan_fingerprint=p["fingerprint"],
             n_frames=len(chosen), n_actions=n, n_positive=positive, mismatches=0,
             original_index=artifact(a.source_index), reference_frames=sources,
             gc6d_physics=gc.provenance, table_supplied_only_for_graspnet_parity=True)
    atomic_json(a.output, signed(r)); return r


def _stem(row, camera):
    return f"scene{row['scene']:06d}_{camera}_img{image_id(row['anno'], camera):06d}"


def verify_entry(e, p):
    verify_artifact(e); verify_artifact(e["audit"]); verify_artifact(e["raw"])
    meta = load_signed(e["audit"]["path"])
    if meta["plan_fingerprint"] != p["fingerprint"] or meta["row"] != e["row"] or meta["bank_sha256"] != e["sha256"]:
        raise ValueError("Frame receipt and frozen plan disagree")
    with np.load(e["path"], allow_pickle=False) as z:
        a = dict(z)
    with np.load(e["raw"]["path"], allow_pickle=False) as z:
        raw = dict(z)
    validate_frame(a, raw["bank"], p["binding"]["tau"])
    expected = quality_counts(raw["friction"].reshape(-1), raw["collision_or_empty"].reshape(-1)).reshape(a["quality_count"].shape)
    if not np.array_equal(expected, a["quality_count"]):
        raise ValueError("Label count disagrees with exact evaluator trace")
    selected = choose_families(raw["native_grasps"][:, 0], p["families"], seed_for(e["row"], p["seed"]))
    if not np.array_equal(selected, a["family_id"]) or not np.array_equal(raw["native_grasps"][selected], raw["bank"][:, 0]):
        raise ValueError("Original candidate sampling changed or used another bank")
    for src in meta["input_files"]:
        verify_artifact(src)
    if meta["quality_audit"]["reference_parity"]["mismatches"] != 0:
        raise ValueError("Failed official singleton check")
    return len(a["family_id"]) * 7


def build(a, source=None, corrector=None, evaluator=None, physics=None):
    """Dependency injection is for CPU tests only; CLI always uses real adapters."""
    from .geometry import action_bank
    from .p62_gc6d_runtime import GC6DSource, FrozenCorrector, ExactGC6D, make_physics
    p = load_signed(a.plan); verify_plan(p)
    injected = any(x is not None for x in (source, corrector, evaluator, physics))
    if injected and not a.smoke:
        raise ValueError("Injected test backend cannot produce a formal manifest")
    if a.limit < 0 or (a.limit and not a.smoke):
        raise ValueError("Partial export requires --smoke and cannot be evaluated formally")
    parity = None if a.smoke else check_parity(a.parity_report, p)
    out = Path(a.output).resolve()
    readonly = [Path(p["dataset_root"]).resolve()] + [Path(x["path"]).parent for x in p["binding"]["assets"].values()]
    for old in readonly:
        if out == old or out in old.parents or old in out.parents:
            raise ValueError("Export root overlaps read-only data/checkpoints")
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".build.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        frozen = out / "plan.json"
        if frozen.exists():
            if load_signed(frozen)["fingerprint"] != p["fingerprint"]:
                raise ValueError("Resume plan changed")
        else:
            atomic_json(frozen, p)
        rows = p["rows"][:a.limit] if a.limit else p["rows"]
        b = p["binding"]
        source = source or GC6DSource(b["config"]["path"], p["dataset_root"], p["camera"], a.device, p["crop_box"])
        corrector = corrector or FrozenCorrector(b["assets"]["depth_corrector"]["path"], a.device)
        evaluator = evaluator or ExactGC6D(p["dataset_root"], p["camera"], a.chunk, a.verify_per_frame)
        physics = physics or make_physics
        if parity is not None and parity["gc6d_physics"] != evaluator.provenance:
            raise ValueError("GC6D physics implementation/config changed after GraspNet parity")
        run_contract = dict(plan_fingerprint=p["fingerprint"], smoke=bool(a.smoke), limit=a.limit,
                            injected_test_backend=injected, evaluator=evaluator.provenance,
                            source=source.provenance, corrected_tf32=False,
                            feature_storage_dtype=b["feature_dtype"], corrector_batch=p["batch_size"])
        run_path = out / "run_contract.json"
        if run_path.exists():
            old = load_signed(run_path); old.pop("fingerprint")
            if old != run_contract:
                raise ValueError("Cannot resume a different runtime/precision/API")
        else:
            atomic_json(run_path, signed(run_contract))
        entries = []
        for start in range(0, len(rows), p["batch_size"]):
            batch_rows = rows[start:start+p["batch_size"]]
            existing = []
            for row in batch_rows:
                receipt = out / "frames" / (_stem(row, p["camera"])+".entry.json")
                if receipt.exists():
                    e = load_signed(receipt); e.pop("fingerprint"); verify_entry(e, p)
                    existing.append(e)
                else:
                    existing.append(None)
            if all(x is not None for x in existing):
                entries += existing; continue
            # Recompute the whole original batch after interruption; never change B4
            # to a partial missing-frame batch or reuse geometry from another plan.
            snapshots, audits = [], []
            for row in batch_rows:
                for src in row["input_files"]:
                    verify_artifact(src)
                snap, audit = source.snapshot(row, p["seed"])
                if len(snap["grasps"]) != 1024 or snap["feature_map"].shape[0] != b["channels"]:
                    raise ValueError("Native source query/feature contract differs")
                snapshots.append(snap); audits.append(audit)
            depths = corrector.predict(snapshots, b["feature_dtype"])
            for row, snap, audit, depth, old_entry in zip(batch_rows, snapshots, audits, depths, existing):
                if old_entry is not None:
                    entries.append(old_entry); continue
                families = choose_families(snap["grasps"][:, 0], p["families"], seed_for(row, p["seed"]))
                bank = action_bank(snap["grasps"][families]); validate_bank(bank)
                g = bank.reshape(-1, 17)
                labels, label_audit = evaluator.evaluate(row, g)
                arrays = physics(snap, depth, bank, b["tau"], b["band"])
                arrays.update(physical_id=physical_ids(g).reshape(bank.shape[:2]), family_id=families,
                              quality_count=quality_counts(labels["friction"], labels["collision_or_empty"]).reshape(bank.shape[:2]))
                validate_frame(arrays, bank, b["tau"])
                stem = _stem(row, p["camera"])
                file = out / "frames" / (stem+".npz")
                rawfile = out / "audit" / (stem+".npz")
                metafile = out / "audit" / (stem+".json")
                # Incomplete receipts fail closed; do not overwrite unproven partial files.
                if any(x.exists() for x in (file, rawfile, metafile)):
                    raise FileExistsError(f"Uncommitted partial frame {stem}; quarantine it before resume")
                atomic_npz(file, arrays)
                atomic_npz(rawfile, dict(bank=bank, native_grasps=snap["grasps"], K=snap["K"],
                    pred_depth=snap["pred_depth"], corrected_depth=depth,
                    friction=labels["friction"], collision_or_empty=labels["collision_or_empty"],
                    empty=labels["empty"], assigned_obj=labels["assigned_obj"]))
                meta = dict(plan_fingerprint=p["fingerprint"], row=row, bank_sha256=sha256(file),
                            input_files=row["input_files"], snapshot_audit=audit, quality_audit=label_audit)
                atomic_json(metafile, signed(meta))
                entry = dict(row=row, **artifact(file), raw=artifact(rawfile), audit=artifact(metafile))
                atomic_json(out/"frames"/(stem+".entry.json"), signed(entry)); entries.append(entry)
            print(f"[GC6D P6-2A] {start+len(batch_rows)}/{len(rows)} frames", flush=True)
        manifest = dict(schema=SCHEMA, generator=GENERATOR,
            partition="smoke" if a.smoke else "independent_holdout", smoke=bool(a.smoke),
            complete=not a.smoke and len(entries) == len(p["rows"]), external_dataset_id="gc6d",
            quality_independently_evaluated=True, clean_geometry_never_model_input=True,
            frozen_model_before_holdout=True, depth_corrector_sha256=b["assets"]["depth_corrector"]["sha256"],
            offsets_m=list(OFFSETS), geometry_tau_m=b["tau"], geometry_band_m=b["band"],
            training_scene_keys=p["lineage"].get("training_scene_keys", []),
            development_scene_keys=p["lineage"].get("development_scene_keys", []),
            official_test_scene_keys=p["lineage"].get("official_test_scene_keys", []),
            plan=artifact(frozen), run_contract=artifact(run_path),
            parity_report=None if parity is None else artifact(a.parity_report),
            n_actions=len(entries)*p["families"]*7, entries=entries,
            note="GC6D official collision/force-closure on ALL exact actions; no NMS/GT background filter/table. Not official AP.")
        dest = out / ("smoke_index.json" if a.smoke else "index.json")
        signed_manifest = signed(manifest)
        if dest.exists() and load_signed(dest) != signed_manifest:
            raise FileExistsError("Completed manifest changed")
        atomic_json(dest, signed_manifest)
        return manifest
    finally:
        lock.unlink()


def audit_manifest(path, recompute_geometry=False, allow_smoke=False):
    m = load_signed(path)
    if m.get("generator") != GENERATOR or m.get("schema") != SCHEMA:
        raise ValueError("Not a GC6D-generated P6-2A cache")
    if not allow_smoke and (m.get("smoke") or not m.get("complete") or m["partition"] != "independent_holdout"):
        raise ValueError("Smoke/partial cache is never a formal holdout")
    verify_artifact(m["plan"]); verify_artifact(m["run_contract"])
    p = load_signed(m["plan"]["path"]); verify_plan(p)
    run = load_signed(m["run_contract"]["path"])
    if run["plan_fingerprint"] != p["fingerprint"] or (not allow_smoke and run["injected_test_backend"]):
        raise ValueError("Cache runtime provenance mismatch")
    if not allow_smoke:
        verify_artifact(m["parity_report"]); check_parity(m["parity_report"]["path"], p)
    if (m["offsets_m"] != list(OFFSETS) or m["geometry_tau_m"] != p["binding"]["tau"]
        or m["geometry_band_m"] != p["binding"]["band"]):
        raise ValueError("External geometry metadata changed")
    expected = p["rows"][:len(m["entries"])] if allow_smoke else p["rows"]
    if [e["row"] for e in m["entries"]] != expected:
        raise ValueError("Missing, reordered or duplicate holdout frames")
    n = 0
    histogram = np.zeros(7, np.int64)
    for e in m["entries"]:
        n += verify_entry(e, p)
        with np.load(e["path"], allow_pickle=False) as z:
            histogram += np.bincount(z["quality_count"].ravel(), minlength=7)
        if recompute_geometry:
            from .p62_gc6d_runtime import make_physics
            with np.load(e["raw"]["path"], allow_pickle=False) as z:
                raw = dict(z)
            reproduced = make_physics(raw, raw["corrected_depth"], raw["bank"], p["binding"]["tau"], p["binding"]["band"])
            with np.load(e["path"], allow_pickle=False) as z:
                for key, value in reproduced.items():
                    if not np.allclose(value, z[key], atol=1e-6, rtol=0):
                        raise ValueError("Stored depth -> token parity failed: "+key)
    if n != m["n_actions"]:
        raise ValueError("Action coverage differs")
    return dict(n_frames=len(m["entries"]), n_actions=n, geometry_recomputed=recompute_geometry,
                formal=not m["smoke"], quality_count_histogram=histogram.tolist(),
                manifest_sha256=sha256(path))


def evaluate(a):
    from .p62_gc6d_runtime import no_tf32
    audit_manifest(a.manifest)
    m = load_signed(a.manifest); p = load_signed(m["plan"]["path"])
    from .p6_independent_eval import evaluate as existing_evaluate
    b = p["binding"]["assets"]
    with no_tf32():
        return existing_evaluate(a.manifest, b["predicted_reader"]["path"], b["corrected_reader"]["path"],
                                 a.output, a.device, a.bootstrap, a.seed, smoke=False)


def lineage_template(a):
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    cfg = json.loads(Path(a.config).read_text())
    paths = dict(base=cfg["base_checkpoint"], selector=cfg["selector_checkpoint"],
                 depth_corrector=a.depth_weight, predicted_reader=a.predicted_weight,
                 corrected_reader=a.corrected_weight)
    result = dict(schema="p62-gc6d-lineage-v1", attested_complete=False, reviewer="",
        reviewed_at="", holdout_used_for_tuning=False, training_scene_keys=[],
        development_scene_keys=[], official_test_scene_keys=[],
        components={k:dict(sha256=sha256(v), gc6d_training_scenes=[], gc6d_development_scenes=[])
                    for k,v in paths.items()},
        review_instructions="Fill prior source scene inventories and GC6D exposure for ALL five frozen components; empty exposure means none, not unknown. Set attested_complete only after review. Never assert unseen objects from scene-only separation.")
    atomic_json(a.output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    t = sub.add_parser("lineage-template")
    for k in ("config", "depth-weight", "predicted-weight", "corrected-weight", "output"):
        t.add_argument("--"+k, required=True)
    t.set_defaults(func=lineage_template)
    t = sub.add_parser("plan")
    for key in ("config", "p1-index", "depth-index", "depth-weight", "predicted-weight", "corrected-weight",
                "lineage", "scenes-json", "dataset-root", "output"):
        t.add_argument("--"+key, required=True)
    t.add_argument("--camera", choices=tuple(CAMERAS), default="realsense-d435")
    t.add_argument("--scene-count", type=int, default=20)
    t.add_argument("--annos", default="0,2,3,5,7,9,10,12")
    t.add_argument("--families", type=int, default=32); t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--seed", type=int, default=0); t.add_argument("--crop-box", nargs=4, type=int)
    t.set_defaults(func=plan)
    t = sub.add_parser("parity-graspnet")
    for k in ("plan", "source-index", "graspnet-root", "output"):
        t.add_argument("--"+k, required=True)
    t.add_argument("--frames", type=int, default=4); t.add_argument("--chunk", type=int, default=64)
    t.set_defaults(func=parity_graspnet)
    t = sub.add_parser("build")
    for k in ("plan", "output"):
        t.add_argument("--"+k, required=True)
    t.add_argument("--parity-report"); t.add_argument("--device", default="cuda:0")
    t.add_argument("--chunk", type=int, default=64); t.add_argument("--verify-per-frame", type=int, default=3)
    t.add_argument("--resume", action="store_true"); t.add_argument("--smoke", action="store_true")
    t.add_argument("--limit", type=int, default=0); t.set_defaults(func=build)
    t = sub.add_parser("audit")
    t.add_argument("--manifest", required=True); t.add_argument("--recompute-geometry", action="store_true")
    t.add_argument("--allow-smoke", action="store_true")
    t.set_defaults(func=lambda a: audit_manifest(a.manifest, a.recompute_geometry, a.allow_smoke))
    t = sub.add_parser("evaluate")
    t.add_argument("--manifest", required=True); t.add_argument("--output", required=True)
    t.add_argument("--device", default="cuda:0"); t.add_argument("--bootstrap", type=int, default=5000)
    t.add_argument("--seed", type=int, default=0); t.set_defaults(func=evaluate)
    a = p.parse_args(); result = a.func(a)
    print(json.dumps({k: v for k, v in result.items() if k.startswith("n_") or k in ("formal", "schema", "partition")}, indent=2))


if __name__ == "__main__":
    main()
