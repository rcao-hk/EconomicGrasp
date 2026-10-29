"""Strict, source-isolated access to 2026-10-27 historical Point/GVAR models.

Run the P1 scripts from an *independent* diagnostic checkout. Only the
model/runtime imports come from --historical-root, whose fingerprint is checked
by the existing rayrope_gvar.runtime.load_model(..., verify=True).

GC6D data are read by the diagnostic branch's loader via an explicit file
import, avoiding silent substitution by a stale historical dataset module.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


P1_REPO = Path(__file__).resolve().parents[1]
LOADED_HISTORICAL_ROOT = None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for data in iter(lambda: src.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def activate_historical(historical_root):
    global LOADED_HISTORICAL_ROOT
    root = Path(historical_root).expanduser().resolve()
    if not (root / "rayrope_gvar/runtime.py").is_file():
        raise FileNotFoundError(f"Historical Point/GVAR runtime absent: {root}")
    if LOADED_HISTORICAL_ROOT is not None and LOADED_HISTORICAL_ROOT != root:
        raise RuntimeError("Cannot mix different historical model roots inside one Python process")
    if LOADED_HISTORICAL_ROOT is None:
        for stem in ("models", "moge_rayrope", "rayrope_gvar", "utils", "dataset"):
            if stem in sys.modules:
                raise RuntimeError(f"Module {stem} loaded before historical source selection")
        sys.path.insert(0, str(root))
        LOADED_HISTORICAL_ROOT = root
    return root


def import_gc6d_dataset():
    """Isolate diagnostic GC6D loader from historical model-root imports."""
    filename = P1_REPO / "dataset" / "gc6d_dataset.py"
    if not filename.is_file():
        raise FileNotFoundError(filename)
    key = "_diagnostic_mixed_p1_gc6d_dataset"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, str(filename))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load GC6D dataloader: {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def load_model(historical_root, checkpoint, device="cuda:0"):
    root = activate_historical(historical_root)
    from rayrope_gvar.runtime import load_model as strict_load
    file = Path(checkpoint).expanduser().resolve()
    if not file.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {file}")
    model, ck = strict_load(str(file), device=device, verify=True)
    if not isinstance(ck, dict) or not isinstance(ck.get("protocol"), dict):
        raise TypeError("Expected historical version/protocol/model checkpoint; not main grasp_design")
    p = ck["protocol"]
    if p.get("partial_run") or int(ck["epoch"]) + 1 != int(p["epochs"]):
        raise ValueError("Require the completed fixed-budget historical checkpoint (not smoke or early stop)")
    if p.get("variant") not in ("rayrope_point", "rayrope_point_gvar"):
        raise ValueError(f"Unsupported historical model: {p.get('variant')}")
    if p.get("code_sha256") is None or p.get("dav2_sha256") is None:
        raise KeyError("Historical checkpoint missing source or DAV2 fingerprint")
    return model, ck, dict(
        checkpoint=str(file),
        checkpoint_sha256=sha256(file),
        historical_root=str(root),
        source_sha256=p["code_sha256"],
        dav2_sha256=p["dav2_sha256"],
        variant=p["variant"],
        completed_epochs=int(p["epochs"]),
        model=p.get("model"),
        depth_supervision=p.get("mixed_depth_supervision"),
    )


def _read_filtered_ids(jsonfile, gcroot):
    from pathlib import Path
    file = Path(jsonfile).expanduser().resolve()
    ids = json.loads(file.read_text(encoding="utf-8"))
    ids = [int(i) for i in ids] if isinstance(ids, list) else None
    if ids is None or len(ids) != 232 or len(set(ids)) != 232:
        raise ValueError(f"Expected 232 unique GC6D filtered training scenes: {file}")
    ids = sorted(ids)
    from _diagnostic_mixed_p1_gc6d_dataset import read_split_scene_ids
    official = set(read_split_scene_ids(gcroot, "train"))
    if set(ids) - official:
        raise ValueError("Filtered file includes GC6D non-training scenes")
    return ids


def gc6d_test_dataset(root, camera="realsense-d435", max_ann=13):
    d = import_gc6d_dataset()
    ds = d.GraspClutter6DMultiDataset(
        root=str(Path(root).expanduser().resolve()),
        camera=camera, split="test", max_ann=max_ann,
        num_points=20000, voxel_size=.005,
        load_label=False, load_gt_supervision=False,
        use_gt_depth=False, use_fuse_depth=False,
        mask_mode="workspace_depth", remove_outlier=None,
        workspace_pose_mode="json", workspace_outlier=.02,
        factor_depth_mode="bop", graspness_mode="instance",
        min_depth=.2, max_depth=1.0, bin_num=256,
        depth_strides=1,
    )
    if len(ds) != len(ds.scene_ids)*13:
        raise ValueError("GC6D expected 13 D435 images per test scene")
    if len(ds.scene_ids) != 235 or len(ds) != 3055:
        raise ValueError(f"Expected official GC6D test 235×13=3055, got {len(ds.scene_ids)}×{len(ds)}")
    return ds


def make_train_sources(gnroot, gntransroot, gcroot, filtered_json, config,
                       gn_label_folder="economic_grasp_label_300views_extend_angle_cdf_depth",
                       gc_label_folder="economic_grasp_label_300views_extend_angle_cdf_depth"):
    """Production source distributions, including per-source depth/graspness.

    GN/Trans are *paired* stride10. GC6D is all 13 D435 frames of 232 scenes.
    """
    if LOADED_HISTORICAL_ROOT is None:
        raise RuntimeError("Call activate_historical/load_model before source dataset construction.")
    from moge_rayrope.mixed import build_bases, paired_frame_indices
    from dataset.cdf_label_adapter import CVAExtendedLabelAdapter
    gc6d = import_gc6d_dataset()
    gnroot = str(Path(gnroot).expanduser().resolve())
    gntransroot = str(Path(gntransroot).expanduser().resolve())
    gcroot = str(Path(gcroot).expanduser().resolve())
    gn, trans = build_bases(gnroot, gntransroot, "train", config, labels=True)
    g_i, t_i, schedule = paired_frame_indices(gn, trans, .1)
    if len(g_i) != 2600 or len(t_i) != 2600:
        raise RuntimeError("GN/Trans must have exactly 2600 matched stride10 frames")
    def wrap(base):
        return CVAExtendedLabelAdapter(
            base, dataset_root=gnroot, use_cdf=True,
            label_folder=gn_label_folder, num_angle=12, num_depth=4,
        )
    ids = _read_filtered_ids(filtered_json, gcroot)
    base = gc6d.GraspClutter6DMultiDataset(
        root=gcroot, camera="realsense-d435", split="train",
        scene_ids=ids, max_ann=13, num_points=20000, voxel_size=.005,
        load_label=True, load_grasp_payload=False,
        use_gt_depth=False, use_fuse_depth=False, graspness_mode="instance",
        mask_mode="workspace_depth", remove_outlier=None,
        workspace_pose_mode="json", workspace_outlier=.02,
        min_depth=config.min_depth, max_depth=config.max_depth,
        bin_num=256, depth_strides=1, extend_angle=True,
        economic_label_root=str(Path(gcroot)/gc_label_folder),
        scene_name_format="plain6", strict_label_check=True,
    )
    if len(base) != 3016:
        raise ValueError(f"Expected filtered GC6D 3016 training frames, got {len(base)}")
    gcd = CVAExtendedLabelAdapter(
        base, dataset_root=gcroot, use_cdf=True,
        label_folder=gc_label_folder, num_angle=12, num_depth=4,
    )
    return dict(
        gn=(Subset(wrap(gn), g_i), gn, g_i),
        gntrans=(Subset(wrap(trans), t_i), trans, t_i),
        gc6d=(gcd, base, list(range(len(gcd)))),
    )


class DeterministicIndexedDataset(Dataset):
    """Stable per-index PyTorch/NumPy/Python RNG, independent of worker count."""
    def __init__(self, dataset, indices, salt=34153):
        self.dataset = dataset
        self.indices = list(indices)
        self.salt = int(salt)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        idx = int(self.indices[row])
        py = random.getstate()
        npy = np.random.get_state()
        rng = torch.get_rng_state()
        try:
            seed = self.salt + idx
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            return self.dataset[idx]
        finally:
            random.setstate(py)
            np.random.set_state(npy)
            torch.set_rng_state(rng)


def sample_source_indices(wrapped, max_frames, seed=0):
    """Choose one random frame per scene first, then additional rounds."""
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    group = {}
    # Source datasets are Subset(CVAAdapter) for paired GN/GNTrans and
    # direct CVAAdapter for GC6D. Index through Subset before scene lookup.
    for i in range(len(wrapped)):
        if isinstance(wrapped, Subset):
            scene = str(wrapped.dataset.scenename[wrapped.indices[i]])
        else:
            scene = str(wrapped.scenename[i])
        group.setdefault(scene, []).append(i)
    rng = np.random.default_rng(seed)
    scenes = sorted(group)
    rng.shuffle(scenes)
    result = []
    for offset in range(max(1, (max_frames + len(scenes)-1)//len(scenes))):
        for scene in scenes:
            if len(result) >= max_frames:
                break
            frames = sorted(group[scene])
            selected = frames[int(rng.integers(0, len(frames)))]
            if selected not in result:
                result.append(selected)
        if len(result)>=max_frames:
            break
    if len(result)<max_frames:
        all_idx=np.arange(len(wrapped))
        rng.shuffle(all_idx)
        result.extend(int(i) for i in all_idx if int(i) not in result)
    return result[:max_frames]
