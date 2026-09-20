"""Rep-A: fixed-action data, deterministic depth interventions and metrics.

No detector, GraspNet API, CAD, or DexNet imports. Labels and actions are never
modified by depth augmentation. Coordinates and depth are in metres.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

VERSION = 1
BASE_MAIN = "52d09f925059bec3643610ecf1f1722894627ee5"
SPLITS = {"train": (0, 100), "test_seen": (100, 130),
          "test_similar": (130, 160), "test_novel": (160, 190)}
VARIANTS = {"A0": (False, False), "A1": (False, True),
            "A2": (True, False), "A3": (True, True)}  # independent RGB, augmentation
THRESHOLDS = np.array([.2, .4, .6, .8, 1., 1.2], np.float32)
DEFAULT_CASES = ("nominal,bias:-5,bias:5,bias:-10,bias:10,bias:-20,bias:20,"
                 "bias:-40,bias:40,scale:-0.02,scale:0.02,scale:-0.05,scale:0.05,"
                 "smooth:5,smooth:10,smooth:20,edge:2")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seed_for(*parts) -> int:
    # Never Python hash(): stable across jobs, process count and launch order.
    return int(digest(parts)[:15], 16) % (2**31 - 1)


def file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


@contextlib.contextmanager
def atomic_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_json(path, value):
    with atomic_file(path) as f:
        f.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def save_npz(path, value):
    with atomic_file(path) as f:
        np.savez_compressed(f, **value)


def save_torch(path, value):
    with atomic_file(path) as f:
        torch.save(value, f)


def load_torch(path):
    # Only use trusted, locally produced model/optimizer checkpoints.
    return torch.load(path, map_location="cpu", weights_only=False)


@contextlib.contextmanager
def exclusive_run(path, *, wait=False):
    """Advisory lock prevents two jobs writing the same output; survives SIGKILL."""
    import fcntl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {path}") from exc
        yield


def list_frames(root, split, limit=0):
    if split not in SPLITS:
        raise ValueError(split)
    lo, hi = SPLITS[split]
    paths = sorted((Path(root) / split).glob("scene_*/ann_*.npz"))
    for p in paths:
        sid = int(p.parent.name.split("_")[-1])
        if not lo <= sid < hi:
            raise ValueError(f"Split leakage: {p} is not in [{lo},{hi})")
    if not paths:
        raise FileNotFoundError(f"No frames: {root}/{split}/scene_*/ann_*.npz")
    return paths[:limit] if limit > 0 else paths


def frame_identity(path):
    p = Path(path)
    return int(p.parent.name.split("_")[-1]), int(p.stem.split("_")[-1])


def fixed_data(path):
    """Read only P0 actions/labels, NOT privileged geometry features."""
    keys = ("actions", "valid", "friction", "utility", "offsets_mm", "zero_index",
            "scene_id", "anno_id", "query_ids", "native_score")
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in keys}
    d["valid"] = d["valid"].astype(bool)
    a, v = d["actions"], d["valid"]
    if a.ndim != 3 or a.shape[-1] != 17 or a.shape[:2] != v.shape:
        raise ValueError(f"Bad P0 action shape: {path}")
    z = int(d["zero_index"])
    if not 0 <= z < len(a) or not np.isclose(d["offsets_mm"][z], 0):
        raise ValueError(f"Bad zero offset: {path}")
    if not v[z].all() or not np.isfinite(a[v]).all():
        raise ValueError(f"Invalid native/actions: {path}")
    # P0 fixes R,w,h,d, unlike the earlier centre re-decoding caches.
    if not np.allclose(a[..., 1:13], a[z:z+1, :, 1:13], atol=1e-7, rtol=0):
        raise ValueError(f"Not a fixed-action Rep-P0 cache: {path}")
    if not np.isfinite(d["friction"][v]).all():
        raise ValueError(f"Unlabelled valid candidates: {path}")
    if d["friction"].shape != v.shape or d["utility"].shape != v.shape:
        raise ValueError(f"Bad labels: {path}")
    y = cdf_targets(d["friction"])
    if not np.allclose(y.mean(-1)[v], d["utility"][v], atol=1e-6):
        raise ValueError(f"Utility/friction mismatch: {path}")
    if tuple(map(int, (d["scene_id"], d["anno_id"]))) != frame_identity(path):
        raise ValueError(f"Scene/frame mismatch: {path}")
    if len(np.unique(d["query_ids"])) != a.shape[1]:
        raise ValueError(f"Duplicate/missing query IDs: {path}")
    return d


def cdf_targets(friction):
    f = np.asarray(friction)
    return (np.isfinite(f)[..., None] & (f[..., None] > 0)
            & (f[..., None] <= THRESHOLDS + 1e-6)).astype(np.float32)


def check_runtime_sources(cache_root):
    """Do not silently use a different upstream reader on an old export."""
    manifest = json.loads((Path(cache_root)/"manifest.json").read_text())
    repo = Path(__file__).resolve().parent
    for rel, expected in manifest.get("upstream_sha256", {}).items():
        if file_sha(repo/rel) != expected:
            raise RuntimeError(f"Upstream code changed: {rel}; use the recorded Rep-A branch")
    return manifest


def read_frame(path, contract=None):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if int(d["version"]) != VERSION:
        raise ValueError(f"Cache version mismatch: {path}")
    if contract is not None and str(d["contract"]) != contract:
        raise ValueError(f"Different Stage-1/export contract: {path}")
    if not np.isfinite(d["image_feature"]).all() or not np.isfinite(d["depth"]).all():
        raise ValueError(f"Non-finite evidence: {path}")
    if array_sha(d["actions"], d["valid"], d["friction"], d["query_ids"]) != str(d["action_sha"]):
        raise ValueError(f"Fixed-action fingerprint mismatch: {path}")
    kq = d["actions"].shape[:2]
    if d["image_feature"].ndim != 3 or d["depth"].ndim != 3 or d["depth"].shape[0] != 1:
        raise ValueError(f"Wrong image/depth shape: {path}")
    h, w = d["depth"].shape[-2:]
    if d["objectness"].shape != (2,h,w) or d["graspness"].shape != (1,h,w):
        raise ValueError(f"Wrong auxiliary-map shape: {path}")
    if d["K"].shape != (3,3) or not np.isfinite(d["K"]).all():
        raise ValueError(f"Bad intrinsics: {path}")
    if d["token_ids"].shape != (kq[1],) or np.any(d["token_ids"] < 0) or np.any(d["token_ids"] >= h*w):
        raise ValueError(f"Bad pixel indices: {path}")
    return d


def tensors(d, device):
    names = ("image_feature", "depth", "K", "objectness", "graspness", "actions", "token_ids")
    out = {k: torch.from_numpy(np.asarray(d[k]).copy()).to(device) for k in names}
    for k in names[:-1]:
        out[k] = out[k].float()
    out["token_ids"] = out["token_ids"].long()
    return out


def parse_case(text):
    if text == "nominal":
        return "nominal", 0.0
    mode, value = text.split(":", 1)
    value = float(value)
    if mode not in ("bias", "scale", "smooth", "edge") or not np.isfinite(value):
        raise ValueError(f"Invalid perturbation: {text}")
    if mode in ("smooth", "edge") and value < 0:
        raise ValueError(text)
    if mode == "scale" and value <= -1:
        raise ValueError(text)
    return mode, value


def training_case(seed, max_bias_mm=20., max_scale=.03, smooth_mm=10., nominal_prob=.25):
    rng = np.random.default_rng(seed)
    if rng.random() < nominal_prob:
        return "nominal"
    mode = rng.choice(["bias", "scale", "smooth"])
    val = (rng.uniform(-max_bias_mm, max_bias_mm) if mode == "bias" else
           rng.uniform(-max_scale, max_scale) if mode == "scale" else
           rng.uniform(0, smooth_mm))
    return f"{mode}:{val:.9g}"


def perturb_depth(depth, case, seed):
    """Perturb estimated observation only. No clipping to model training bounds.

    Bias uses mm, scale is fractional, smooth uses RMS mm, edge is a depth-only
    image shift in pixels. Invalid input remains invalid. Physical candidate
    actions, RGB, K and ground-truth labels are not arguments to this function.
    """
    mode, value = parse_case(case)
    if mode == "nominal" or value == 0:
        return depth.clone(), {"depth_rms_mm": 0., "depth_bias_mm": 0., "clamped_fraction": 0.}
    original = depth.float()
    valid = torch.isfinite(original) & (original > 0)
    if not bool(valid.any()):
        raise ValueError("Depth map has no valid pixels")
    work = torch.where(valid, original, torch.zeros_like(original))
    h, w = original.shape[-2:]
    if mode == "bias":
        changed = work + value / 1000.
    elif mode == "scale":
        changed = work * (1. + value)
    elif mode == "smooth":
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        coarse = torch.randn((1, 1, 5, 5), generator=gen).to(depth.device)
        field = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)[0]
        field = field - field[valid].mean()
        field = field / field[valid].square().mean().sqrt().clamp_min(1e-8)
        changed = work + field * (value / 1000.)
    else:
        sign = 1 if seed % 2 else -1
        yy, xx = torch.meshgrid(torch.arange(h, device=depth.device),
                                torch.arange(w, device=depth.device), indexing="ij")
        grid = torch.stack((2*(xx+.5+sign*value)/w-1, 2*(yy+.5)/h-1), -1).float()[None]
        changed = F.grid_sample(work[None], grid, padding_mode="border", align_corners=False)[0]
    clamp = (changed < .01) & valid
    changed = torch.where(valid, changed.clamp_min(.01), original)
    delta = (changed - original)[valid]
    if not delta.numel():
        raise ValueError("Depth map has no valid pixels")
    return changed, {"depth_rms_mm": float(delta.square().mean().sqrt())*1000,
                     "depth_bias_mm": float(delta.mean())*1000,
                     "clamped_fraction": float(clamp.float().mean())}


def choose(pred_u, valid, zero, margin):
    scores = np.where(valid, pred_u, -np.inf).copy()
    scores[zero] = -np.inf
    alt = scores.argmax(0)
    ids = np.arange(len(alt))
    has_alt = np.isfinite(scores).any(0)
    selected = np.where(has_alt & (scores[alt, ids] > pred_u[zero] + margin), alt, zero)
    return selected.astype(np.int64)


def metrics(prob, d, margin):
    prob = np.asarray(prob, np.float64)
    valid = d["valid"].astype(bool)
    u = d["utility"].astype(float)
    z = int(d["zero_index"])
    y = cdf_targets(d["friction"])
    p = np.clip(prob, 1e-7, 1-1e-7)
    pu = p.mean(-1)
    selected = choose(pu, valid, z, margin)
    q = np.arange(u.shape[1])
    su, nu = u[selected, q], u[z]
    oracle = np.where(valid, u, -np.inf).max(0)
    ss, ns = y[selected, q, 3], y[z, :, 3]
    # Within-ray comparisons, excluding exact-utility ties and invalid pairs.
    du = u[:, None, :] - u[None, :, :]
    dp = pu[:, None, :] - pu[None, :, :]
    pairs = (np.triu(np.ones((len(u), len(u)), bool), 1)[..., None]
             & valid[:, None, :] & valid[None, :, :] & (np.abs(du) > 1e-7))
    pair_correct = np.where(np.abs(dp) <= 1e-12, .5, (dp * du > 0).astype(float))
    result = {"num_queries": u.shape[1], "num_actions": int(valid.sum()),
              "native_utility": float(nu.mean()), "selected_utility": float(su.mean()),
              "oracle_utility": float(oracle.mean()), "utility_gain": float((su-nu).mean()),
              "success08": float(ss.mean()), "native_success08": float(ns.mean()),
              "success08_gain": float((ss-ns).mean()),
              "rescue08": float(((ns==0)&(ss==1)).mean()),
              "harm08": float(((ns==1)&(ss==0)).mean()),
              "move_rate": float((selected != z).mean()),
              "cdf_bce": float(-(y*np.log(p)+(1-y)*np.log(1-p))[valid].mean()),
              "brier": float(((p-y)**2)[valid].mean()),
              "pair_correct": float(pair_correct[pairs].sum()), "pair_count": int(pairs.sum())}
    return result, selected


def aggregate(rows):
    if not rows:
        raise ValueError("No result rows")
    total_q = sum(r["num_queries"] for r in rows)
    total_a = sum(r["num_actions"] for r in rows)
    cols = ("native_utility", "selected_utility", "oracle_utility", "utility_gain", "success08",
            "native_success08", "success08_gain", "rescue08", "harm08", "move_rate")
    out = {k: sum(r[k]*r["num_queries"] for r in rows)/total_q for k in cols}
    for k in ("cdf_bce", "brier"):
        out[k] = sum(r[k]*r["num_actions"] for r in rows)/total_a
    pairs = sum(r["pair_count"] for r in rows)
    out["within_ray_pair_accuracy"] = sum(r["pair_correct"] for r in rows)/pairs if pairs else None
    h = out["oracle_utility"] - out["native_utility"]
    out["headroom_recovery"] = out["utility_gain"]/h if h > 1e-9 else None
    out.update(num_frames=len(rows), num_queries=total_q, num_actions=total_a)
    return out


def tune_margin(predictions, frames):
    rows = []
    for margin in np.linspace(0, 1, 51):  # 1 guarantees native fallback
        agg = aggregate([metrics(p, f, margin)[0] for p, f in zip(predictions, frames)])
        rows.append({"margin": float(margin), **agg})
    best = max(rows, key=lambda r: (r["selected_utility"], -r["harm08"], -r["move_rate"]))
    return best, rows
