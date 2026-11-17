"""GC6D -> P6-2A cache contracts; no CUDA, dataset APIs or CAD imports.

Hashes are integrity/provenance checks, not proof of scientific independence.
The manifest keeps labels separate from the two model-input geometry arrays.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "shared-visible-geometry-p6-external-bank-v1"
GENERATOR = "gc6d-p62-v1"
CAMERAS = {"realsense-d415": 1, "realsense-d435": 2, "azure-kinect": 3, "zivid": 4}
OFFSETS = (0., -.005, .005, -.010, .010, -.020, .020)
FIELDS = {"predicted_physics", "corrected_physics", "local_xyz", "bank_dims",
          "physical_id", "family_id", "quality_count"}
FRICTIONS = np.array([.2, .4, .6, .8, 1., 1.2], dtype=np.float64)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def signed(obj: dict) -> dict:
    if "fingerprint" in obj:
        raise ValueError("Already signed")
    return dict(obj, fingerprint=digest(obj))


def load_signed(path: str | Path) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = obj.pop("fingerprint", None)
    if expected is None or digest(obj) != expected:
        raise ValueError(f"Signature mismatch: {path}")
    return dict(obj, fingerprint=expected)


def atomic_json(path: str | Path, obj: dict) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, p)


def atomic_npz(path: str | Path, arrays: dict) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, p)


def image_id(anno: int, camera: str) -> int:
    if camera not in CAMERAS or not isinstance(anno, (int, np.integer)) or not 0 <= anno < 13:
        raise ValueError("GC6D requires ann_id 0..12 and an explicit camera")
    return int(anno) * 4 + CAMERAS[camera]


def physical_ids(grasps: np.ndarray) -> np.ndarray:
    """Byte-identical to interactive_grasp.contracts.physical_ids (no scores)."""
    g = np.asarray(grasps, dtype="<f4")
    if g.ndim != 2 or g.shape[1] != 17 or not np.isfinite(g).all():
        raise ValueError("Expected finite N x 17 grasps")
    return np.asarray([hashlib.sha256(x[1:16].tobytes()).hexdigest() for x in g], dtype="<U64")


def seed_for(row: dict, seed: int) -> int:
    key = f"{row['domain']}_scene{int(row['scene']):04d}_frame{int(row['anno']):04d}"
    return int(digest([key, int(seed)])[:8], 16)


def choose_families(scores: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Same half-source-top128/half-remainder sampling as shared_geometry.data."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 1 <= n <= len(scores):
        raise ValueError("Invalid source scores/family budget")
    rng = np.random.default_rng(seed)
    high = np.argsort(-scores, kind="stable")[:min(128, len(scores))]
    chosen = rng.choice(high, size=min(n // 2, len(high)), replace=False)
    rest = np.setdiff1d(np.arange(len(scores)), chosen)
    chosen = np.r_[chosen, rng.choice(rest, size=n-len(chosen), replace=False)]
    return np.sort(chosen).astype(np.int64)


def validate_bank(bank: np.ndarray) -> None:
    g = np.asarray(bank)
    if g.dtype != np.float32 or g.ndim != 3 or g.shape[1:] != (7, 17) or len(g) < 1:
        raise ValueError("Expected float32 F x 7 x 17 physical bank")
    if not np.isfinite(g).all() or np.any(g[..., 15] <= 0):
        raise ValueError("Nonfinite bank or action behind camera")
    if np.any((g[..., 1] < 0) | (g[..., 1] > .100001)):
        raise ValueError("P5 source width cap is 0.10 m, not GC6D's AP width cap")
    if not np.allclose(g[..., 2], .02, atol=1e-6, rtol=0):
        raise ValueError("Physical height must match the 0.02 m evaluator gripper")
    r = g[..., 4:13].reshape(-1, 3, 3)
    if not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-3) or np.any(np.linalg.det(r) < .999):
        raise ValueError("Bad grasp rotation")
    native = g[:, 0]
    expected = np.repeat(native[:, None], 7, axis=1)
    direction = native[:, 13:16] / np.maximum(np.linalg.norm(native[:, 13:16], axis=1, keepdims=True), 1e-8)
    expected[:, :, 13:16] += direction[:, None] * np.asarray(OFFSETS, np.float32)[None, :, None]
    if not np.array_equal(g, expected):
        raise ValueError("Bank does not match fixed ordered Euclidean camera-ray offsets")


def quality_counts(friction: np.ndarray, collision: np.ndarray) -> np.ndarray:
    f = np.round(np.asarray(friction, dtype=np.float64), 2)
    col = np.asarray(collision)
    if f.ndim != 1 or col.shape != f.shape or col.dtype != np.bool_ or not np.isfinite(f).all():
        raise ValueError("Bad exact quality result")
    if not np.isin(f, np.r_[-1., FRICTIONS]).all():
        raise ValueError("Friction must be -1 or a canonical six-grid value")
    return ((f[:, None] > 0) & (f[:, None] <= FRICTIONS) & ~col[:, None]).sum(-1).astype(np.int64)


def validate_frame(a: dict, bank: np.ndarray | None = None, tau: float = .03) -> None:
    if set(a) != FIELDS:
        raise ValueError("P6-2A NPZ must contain exactly the seven approved arrays")
    p, c, q = a["predicted_physics"], a["corrected_physics"], a["quality_count"]
    if p.ndim != 4 or p.shape[1:] != (7, 75, 6) or c.shape != p.shape:
        raise ValueError("Bad geometry shape")
    n = len(p)
    shapes = {"local_xyz": (n, 7, 75, 3), "bank_dims": (n, 7, 3),
              "physical_id": (n, 7), "family_id": (n,), "quality_count": (n, 7)}
    if n < 1 or any(a[k].shape != v for k, v in shapes.items()):
        raise ValueError("Physical arrays misaligned")
    if q.dtype.kind not in "iu" or np.any((q < 0) | (q > 6)):
        raise ValueError("Exact count must be integer 0..6")
    if a["family_id"].dtype.kind not in "iu" or len(set(a["family_id"].tolist())) != n:
        raise ValueError("Family IDs must be unique integer source-query indices")
    ids = a["physical_id"].astype(str)
    if len(set(ids.ravel())) != n * 7:
        raise ValueError("Duplicate physical actions: do not deduplicate or relabel silently")
    for x in (p, c, a["local_xyz"], a["bank_dims"]):
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("Geometric inputs must be finite float32")
    for x in (p, c):
        if (np.any((x[..., 0] < 0) | (x[..., 0] > 1)) or
            np.any(np.abs(x[..., 1]) > 1) or
            not np.isin(x[..., 2:], (0., 1.)).all() or
            not np.all(x[..., 2:].sum(-1) == 1)):
            raise ValueError("Bad normalized UDF/projective/one-hot observation states")
    if not 0 < tau <= .2:
        raise ValueError("Bad geometry scale")
    if bank is not None:
        validate_bank(bank)
        if bank.shape[:2] != q.shape or not np.array_equal(physical_ids(bank.reshape(-1, 17)).reshape(q.shape), ids):
            raise ValueError("Physical IDs do not hash the actual 17D poses")
        if not np.array_equal(a["bank_dims"], bank[..., 1:4]):
            raise ValueError("bank_dims must be width,height,insertion in metres")
        axes = np.meshgrid(np.linspace(0, 1, 5, dtype=np.float32),
                           np.linspace(0, 1, 5, dtype=np.float32),
                           np.linspace(0, 1, 3, dtype=np.float32), indexing="ij")
        u = np.stack(axes, -1).reshape(75, 3)
        local = np.empty((*bank.shape[:2], 75, 3), np.float32)
        local[..., 0] = -.03 + (bank[..., 3, None] + .03) * u[:, 0]
        local[..., 1] = (bank[..., 1, None] + .02) * (u[:, 1] - .5)
        local[..., 2] = bank[..., 2, None] * (u[:, 2] - .5)
        if not np.allclose(local, a["local_xyz"], atol=1e-7, rtol=0):
            raise ValueError("Support stencil changed")


def check_lineage(lineage: dict, assets: dict, scenes: list[int]) -> None:
    """Complete lineage is explicitly attested; no test split masquerading as dev."""
    if lineage.get("schema") != "p62-gc6d-lineage-v1" or lineage.get("attested_complete") is not True:
        raise ValueError("Provide a complete, reviewed checkpoint lineage inventory")
    if not lineage.get("reviewer") or not lineage.get("reviewed_at"):
        raise ValueError("Lineage needs reviewer and review timestamp")
    for key in ("training_scene_keys", "development_scene_keys", "official_test_scene_keys"):
        if key not in lineage or not isinstance(lineage[key], list):
            raise ValueError("Missing namespace-qualified source inventory: "+key)
        for value in lineage[key]:
            if (not isinstance(value, list) or len(value) != 2 or
                not isinstance(value[0], str) or not isinstance(value[1], int)):
                raise ValueError("Scene inventory entries must be [dataset_namespace, integer_id]")
            if value[0].lower() in ("gc6d", "graspclutter6d") and value[1] in scenes:
                raise ValueError("Holdout overlaps prior scene inventory")
    components = lineage.get("components", {})
    if set(components) != set(assets):
        raise ValueError("Lineage must cover base, selector, depth_corrector and both readers")
    for name, asset in assets.items():
        entry = components[name]
        if entry.get("sha256") != asset["sha256"]:
            raise ValueError(f"Lineage checkpoint hash mismatch: {name}")
        if "gc6d_training_scenes" not in entry or "gc6d_development_scenes" not in entry:
            raise ValueError(f"Missing GC6D exposure inventory for {name}")
        exposure = entry["gc6d_training_scenes"] + entry["gc6d_development_scenes"]
        if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in exposure):
            raise ValueError("Exposure must be explicit integer scene IDs")
        if set(exposure) & set(scenes):
            raise ValueError(f"Holdout exposure in {name}")
    if lineage.get("holdout_used_for_tuning") is not False:
        raise ValueError("An independent holdout must not have been used for tuning")


def verify_artifact(entry: dict) -> None:
    if sha256(entry["path"]) != entry["sha256"]:
        raise ValueError(f"Artifact changed: {entry['path']}")


def cad_paths(root: str | Path, obj_id: int) -> list[Path]:
    """Match the official trusted DexNet cache preference; never create a cache."""
    root = Path(root)
    ply = root / "models_m" / f"obj_{obj_id:06d}.ply"
    cache = root / "dex_models" / f"obj_{obj_id:06d}.pkl"
    if cache.is_file():
        return [ply, cache]
    stem = root / "models_obj_m" / f"obj_{obj_id:06d}"
    return [ply, Path(str(stem)+".obj"), Path(str(stem)+".sdf")]
