"""Shared runtime for online DAV2 Metric Grasp Field experiments.

Pure utilities stay importable without the repository's CUDA extensions.
Heavy dataset/model imports occur only after command-line parsing.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import Subset

from metric_grasp_field_core import MetricFieldConfig

BASE_MAIN_SHA = "52d09f925059bec3643610ecf1f1722894627ee5"
VERSION = "dav2_metric_grasp_field_detach_v1"
DEFAULT_WORK = "/data2/robotarm/result/grasp/rgbgrasp/dav2_metric_grasp_field_10pct"
BASE_CONFIG = {
    "num_view": 300, "num_angle": 12, "num_depth": 4, "m_point": 1024,
    "grasp_max_width": .1, "graspness_threshold": .1,
    "objectness_loss_weight": 1., "graspness_loss_weight": 10.,
    "view_loss_weight": 100., "score_loss_weight": 1.,
    "width_loss_weight": 10., "depth_prob_loss_weight": 10.,
    "kview_mode": "A1", "kview_k": 1, "kview_tau": 1.,
    "kview_sample_from": "minmax_norm", "kview_patch_size": 6,
    "kview_metric_radius": .08, "kview_radius_px_min": 8., "kview_radius_px_max": 64.,
    "kview_group_dim": 256, "kview_group_heads": 4, "kview_group_dropout": .05,
    "kview_group_chunk": 256, "kview_head_dim": 128, "kview_head_hidden_dim": 64,
    "kview_head_layers": 2, "kview_head_heads": 4, "kview_attn_dropout": .05,
    "kview_head_dropout": .15, "num_cdf_thresholds": 6, "cdf_increment_bias": -4.,
    "multi_modal": True, "use_cdf": True, "extend_angle": True,
    "use_obs_depth": False, "use_gt_depth": False, "use_fuse_depth": False,
    "use_top4_view_infer": False, "graspness_mode": "scene",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_fingerprint():
    root = Path(__file__).resolve().parent
    paths = ["metric_field_runtime.py", "metric_grasp_field_core.py",
             "models/economicgrasp_metric_field.py", "train_metric_grasp_field.py",
             "inference_metric_grasp_field.py", "eval_metric_grasp_field.py",
             "models/economicgrasp_bip3d.py", "models/economicgrasp_depth.py",
             "models/dinov2_dpt.py", "models/kview_query_transformer.py",
             "models/loss_economicgrasp_depth_kview_transformer.py",
             "utils/label_generation.py", "utils/arguments.py", "dataset/graspnet_dataset.py"]
    return digest({p: sha256_file(root/p) for p in paths})


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def ensure_manifest(path, payload):
    """Identical concurrent inference shards may share one manifest."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(path.suffix + ".lock"), "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != payload:
                raise RuntimeError("Existing output protocol changed; use a new output root")
        else:
            atomic_json(path, payload)


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def sampled_indices(total, fraction=.1, max_frames=0):
    """Same convention as main: all scenes, frame stride round(1/fraction).

    fraction=.1 -> ann 0,10,...,250 (26/256, approximately 10%, not 10 scenes).
    Only reciprocal-integer fractions are accepted so requested/actual sampling
    cannot silently diverge. max_frames is for separately labelled smoke runs.
    """
    if total < 0 or max_frames < 0 or not 0 < fraction <= 1:
        raise ValueError("Invalid total/fraction/frame cap")
    stride = round(1/fraction)
    if abs(fraction - 1/stride) > 1e-8:
        raise ValueError("Use reciprocal-integer fractions, e.g. 1, .5, .1, .05")
    result = [i for start in range(0, total, 256)
              for i in range(start, min(start+256, total), stride)]
    return result[:max_frames] if max_frames else result


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda_current": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda_current"] is not None:
        torch.cuda.set_rng_state(state["cuda_current"])


def configure_base(config, field_config, pose_mode, *, top4=False):
    # utils.arguments parses at import time. Do NOT let it parse our own CLI.
    sys.argv = [sys.argv[0]]
    from utils.arguments import cfgs
    for key, value in config.items():
        setattr(cfgs, key, value)
    cfgs.min_depth, cfgs.max_depth = field_config.min_depth, field_config.max_depth
    cfgs.bin_num = 256  # main metric head compatibility; field bins are separate.
    cfgs.pose_depth_mode = pose_mode
    cfgs.use_top4_view_infer = bool(top4)
    return cfgs


def make_dataset(root, split, fraction, *, labels, max_frames=0):
    from dataset.graspnet_dataset import GraspNetMultiDataset
    dataset = GraspNetMultiDataset(
        root, camera="realsense", split=split, num_points=20000,
        remove_outlier=True, augment=False, load_label=labels,
        use_gt_depth=False, use_fuse_depth=False, graspness_mode="scene",
        min_depth=.2, max_depth=1., bin_num=256, depth_strides=1,
        extend_angle=True,
    )
    indices = sampled_indices(len(dataset), fraction, max_frames)
    return dataset, Subset(dataset, indices), indices


def move_batch(batch, device, *, inference=False):
    """Variable per-object CDF/width annotations MUST remain CPU-resident.

    Main's label matcher selects only required rows and transfers those itself.
    At inference, remove all supervision/sensor depth tensors from the forward
    input. This preserves main's dataset crop; it does NOT claim that upstream
    dataset cropping/workspace generation is itself annotation-independent.
    """
    out = {}
    for key, value in batch.items():
        if isinstance(value, (list, tuple)):
            if not inference:
                out[key] = value
            continue
        if inference and (key.startswith("gt_") or key.startswith("sensor_") or
                          key.startswith("objectness_label") or key.startswith("graspness_label") or
                          key.startswith("depth_prob") or key == "cdf_thresholds"):
            continue
        if key in ("point_clouds", "cloud_colors", "coordinates_for_voxel"):
            continue
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def dataset_schedule(dataset, indices):
    names = dataset.scene_list()
    return [[int(names[i].split("_")[-1]), int(i % 256)] for i in indices]


def construct_model(protocol, *, initialize=False, device="cpu", top4=False):
    field = MetricFieldConfig(**protocol["field"])
    configure_base(protocol["base_config"], field, protocol["pose_mode"], top4=top4)
    from models.economicgrasp_metric_field import EconomicGraspMetricField
    return EconomicGraspMetricField(
        field, encoder=protocol["encoder"], pose_depth_mode=protocol["pose_mode"],
        seed_selection_mode=protocol["seed_mode"],
        init_checkpoint=protocol.get("init_checkpoint", "") if initialize else "",
    ).to(device)
