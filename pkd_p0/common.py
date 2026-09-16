"""Common utilities for PKD P0 diagnostics on the current EconomicGrasp code.

The package intentionally targets the corrected current CVA-CDF path. It does
not contain compatibility branches for the old observed-depth, explicit
angle/depth-head, or learned-collision-head checkpoints.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import inspect
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch


CURRENT_CONTRACT_VERSION = 2
CDF_THRESHOLDS: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
SPLIT_SCENE_RANGES: Dict[str, Tuple[int, int]] = {
    "train": (0, 99),
    "test_seen": (100, 129),
    "test_similar": (130, 159),
    "test_novel": (160, 189),
}


class ContractError(RuntimeError):
    """Raised when an experiment no longer represents the intended protocol."""


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(payload: Mapping[str, Any], path: os.PathLike[str] | str) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def atomic_npz_dump(path: os.PathLike[str] | str, *, compress: bool = False, **arrays: np.ndarray) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    saver = np.savez_compressed if compress else np.savez
    with open(tmp, "wb") as handle:
        saver(handle, **arrays)
    os.replace(tmp, path)


def atomic_torch_save(payload: Mapping[str, Any], path: os.PathLike[str] | str) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def clone_batch(batch: Mapping[str, Any]) -> Dict[str, Any]:
    """Clone a model-input mapping without copying immutable metadata deeply."""
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.clone()
        elif isinstance(value, np.ndarray):
            out[key] = value.copy()
        elif isinstance(value, list):
            out[key] = [item.clone() if torch.is_tensor(item) else copy.deepcopy(item) for item in value]
        elif isinstance(value, tuple):
            out[key] = tuple(item.clone() if torch.is_tensor(item) else copy.deepcopy(item) for item in value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def move_tensors(batch: MutableMapping[str, Any], device: torch.device, *, non_blocking: bool = True) -> MutableMapping[str, Any]:
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=non_blocking)
        elif isinstance(value, list):
            batch[key] = [item.to(device, non_blocking=non_blocking) if torch.is_tensor(item) else item for item in value]
    return batch


def parse_csv_ints(text: str) -> List[int]:
    return [int(token.strip()) for token in str(text).split(",") if token.strip()]


def scene_ids_for_split(split: str, explicit: str = "") -> List[int]:
    if split not in SPLIT_SCENE_RANGES:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLIT_SCENE_RANGES)}")
    lo, hi = SPLIT_SCENE_RANGES[split]
    ids = parse_csv_ints(explicit) if explicit else list(range(lo, hi + 1))
    invalid = [scene_id for scene_id in ids if not lo <= scene_id <= hi]
    if invalid:
        raise ValueError(f"Split {split!r} covers scenes {lo}-{hi}; invalid ids: {invalid}")
    return ids


def annotation_ids(sample_interval: float) -> List[int]:
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    stride = 1 if sample_interval >= 1.0 else max(1, int(round(1.0 / sample_interval)))
    return list(range(0, 256, stride))


def filtered_kwargs(callable_obj: Any, kwargs: Mapping[str, Any], *, require: Sequence[str] = ()) -> Dict[str, Any]:
    """Filter keyword arguments against a runtime signature.

    This keeps the diagnostic overlay tolerant to harmless constructor additions
    in the repository while still failing when a required diagnostic knob is
    unavailable.
    """
    signature = inspect.signature(callable_obj)
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    if accepts_var_kw:
        selected = dict(kwargs)
    else:
        selected = {key: value for key, value in kwargs.items() if key in signature.parameters}
    missing = [key for key in require if key not in selected]
    if missing:
        raise ContractError(
            f"{getattr(callable_obj, '__qualname__', callable_obj)!r} does not accept required arguments {missing}; "
            f"available parameters are {list(signature.parameters)}"
        )
    return selected


def parse_before_repo_imports(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse our CLI before EconomicGrasp's import-time parser sees argv."""
    args = parser.parse_args()
    sys.argv[:] = [sys.argv[0]]
    return args


@dataclass(frozen=True)
class CheckpointContract:
    path: str
    sha256: str
    distill_stage: int
    contract_version: int
    seed_selection_mode: str
    geometry_depth_source: str
    pose_depth_mode: str
    use_fuse_depth: bool
    depth_head_executed: bool
    legacy_dataset_use_gt_depth: bool
    cdf_weight_key: str
    cdf_bias_key: str
    feature_dim: int
    num_depths: int
    num_thresholds: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _one_suffix_key(state: Mapping[str, Any], suffix: str) -> str:
    matches = [str(key) for key in state if str(key).endswith(suffix)]
    if len(matches) != 1:
        raise ContractError(f"Expected exactly one state key ending in {suffix!r}; found {matches}")
    return matches[0]


def load_current_checkpoint(
    path: os.PathLike[str] | str,
    *,
    expected_stage: Optional[int] = None,
    allow_stage0_geometry_gt: bool = True,
) -> Tuple[Mapping[str, Any], CheckpointContract]:
    path = str(Path(path).expanduser().resolve())
    loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, Mapping) or "model_state_dict" not in loaded:
        raise ContractError(f"{path} is not a full EconomicGrasp checkpoint")
    state = loaded["model_state_dict"]
    if not isinstance(state, Mapping):
        raise ContractError("model_state_dict is not a mapping")

    required = (
        "distill_stage",
        "distill_contract_version",
        "seed_selection_mode",
        "geometry_depth_source",
        "pose_depth_mode",
        "use_fuse_depth",
        "depth_head_executed",
        "legacy_dataset_use_gt_depth",
    )
    missing = [key for key in required if key not in loaded]
    if missing:
        raise ContractError(f"Checkpoint predates the corrected distillation contract; missing {missing}")

    stage = int(loaded["distill_stage"])
    if expected_stage is not None and stage != int(expected_stage):
        raise ContractError(f"Expected distill_stage={expected_stage}, got {stage}")
    version = int(loaded["distill_contract_version"])
    if version != CURRENT_CONTRACT_VERSION:
        raise ContractError(f"Expected distill_contract_version={CURRENT_CONTRACT_VERSION}, got {version}")
    if str(loaded["seed_selection_mode"]) != "image_fps":
        raise ContractError(f"P0 diagnostics require image_fps, got {loaded['seed_selection_mode']!r}")
    if bool(loaded["legacy_dataset_use_gt_depth"]):
        raise ContractError("Legacy dataset --use_gt_depth checkpoint is not accepted")

    geometry_source = str(loaded["geometry_depth_source"])
    if stage == 1 and geometry_source != "pred":
        raise ContractError(f"Stage-1 checkpoint must use predicted geometry, got {geometry_source!r}")
    if stage == 0 and not allow_stage0_geometry_gt and geometry_source != "pred":
        raise ContractError(f"Unexpected Stage-0 geometry source {geometry_source!r}")

    legacy_suffixes = (
        "kview_grasp_module.decoder.score_head.weight",
        "kview_grasp_module.decoder.depth_head.weight",
        "kview_grasp_module.decoder.collision_head.weight",
    )
    present_legacy = [suffix for suffix in legacy_suffixes if any(str(k).endswith(suffix) for k in state)]
    if present_legacy:
        raise ContractError(f"Legacy decoder heads are not supported: {present_legacy}")

    cdf_weight_key = _one_suffix_key(state, "kview_grasp_module.decoder.cdf_head.weight")
    cdf_bias_key = _one_suffix_key(state, "kview_grasp_module.decoder.cdf_head.bias")
    weight = state[cdf_weight_key]
    if not torch.is_tensor(weight) or weight.ndim != 3 or int(weight.shape[-1]) != 1:
        raise ContractError(f"Unexpected CDF head shape: {getattr(weight, 'shape', None)}")
    num_thresholds = len(CDF_THRESHOLDS)
    if int(weight.shape[0]) % num_thresholds:
        raise ContractError(f"CDF output channels {weight.shape[0]} are not divisible by {num_thresholds}")

    metadata = {
        key: to_jsonable(value)
        for key, value in loaded.items()
        if key not in {"model_state_dict", "optimizer_state_dict"}
    }
    contract = CheckpointContract(
        path=path,
        sha256=sha256_file(path),
        distill_stage=stage,
        contract_version=version,
        seed_selection_mode=str(loaded["seed_selection_mode"]),
        geometry_depth_source=geometry_source,
        pose_depth_mode=str(loaded["pose_depth_mode"]),
        use_fuse_depth=bool(loaded["use_fuse_depth"]),
        depth_head_executed=bool(loaded["depth_head_executed"]),
        legacy_dataset_use_gt_depth=bool(loaded["legacy_dataset_use_gt_depth"]),
        cdf_weight_key=cdf_weight_key,
        cdf_bias_key=cdf_bias_key,
        feature_dim=int(weight.shape[1]),
        num_depths=int(weight.shape[0] // num_thresholds),
        num_thresholds=num_thresholds,
        metadata=metadata,
    )
    return loaded, contract


def resolve_tensor(
    mappings: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
    *,
    required: bool = True,
    ndim: Optional[Sequence[int] | int] = None,
) -> Tuple[Optional[str], Optional[torch.Tensor]]:
    allowed_ndim = None if ndim is None else ({ndim} if isinstance(ndim, int) else set(ndim))
    for mapping in mappings:
        for key in aliases:
            value = mapping.get(key)
            if torch.is_tensor(value) and (allowed_ndim is None or value.ndim in allowed_ndim):
                return key, value
    # suffix/normalized fallback
    normalized_aliases = {re.sub(r"[^a-z0-9]", "", alias.lower()) for alias in aliases}
    for mapping in mappings:
        for key, value in mapping.items():
            if not torch.is_tensor(value):
                continue
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized_key in normalized_aliases and (allowed_ndim is None or value.ndim in allowed_ndim):
                return str(key), value
    if required:
        available = sorted({str(key) for mapping in mappings for key, value in mapping.items() if torch.is_tensor(value)})
        raise KeyError(f"None of tensor aliases {list(aliases)} were found. Tensor keys: {available[:200]}")
    return None, None


def friction_to_cdf(friction: torch.Tensor, thresholds: Sequence[float] = CDF_THRESHOLDS) -> torch.Tensor:
    values = friction.float()
    threshold_tensor = values.new_tensor(tuple(float(x) for x in thresholds))
    return ((values.unsqueeze(-1) > 0.0) & (values.unsqueeze(-1) <= threshold_tensor)).float()


def cdf_utility_from_logits(logits: torch.Tensor, threshold_dim: int = -1) -> torch.Tensor:
    return torch.sigmoid(logits.float()).mean(dim=threshold_dim)


def binary_entropy(probability: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = probability.float().clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def binary_cross_entropy_prob(probability: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = probability.float().clamp(eps, 1.0 - eps)
    y = target.float()
    return -(y * p.log() + (1.0 - y) * (1.0 - p).log())


def rotation_geodesic_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> np.ndarray:
    """Pairwise row-aligned SO(3) geodesic distance in degrees."""
    relative = np.einsum("nij,njk->nik", np.transpose(rotation_a, (0, 2, 1)), rotation_b)
    trace = np.trace(relative, axis1=1, axis2=2)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def split_from_scene(scene_id: int) -> str:
    for split, (lo, hi) in SPLIT_SCENE_RANGES.items():
        if lo <= int(scene_id) <= hi:
            return split
    raise ValueError(f"Scene id {scene_id} is outside GraspNet-1Billion 0-189")


def parse_scene_anno_from_path(path: os.PathLike[str] | str) -> Tuple[int, int]:
    text = str(path)
    scene_match = re.search(r"scene[_-]?(\d{4})", text)
    if scene_match is None:
        raise ValueError(f"Cannot infer scene id from {path}")
    filename = Path(path).stem
    anno_match = re.search(r"(?:ann[_-]?)?(\d{4})(?:\.p0_candidates)?$", filename)
    if anno_match is None:
        raise ValueError(f"Cannot infer annotation id from {path}")
    return int(scene_match.group(1)), int(anno_match.group(1))


@contextlib.contextmanager
def temporary_attributes(obj: Any, **updates: Any) -> Iterator[None]:
    previous: Dict[str, Any] = {}
    missing: List[str] = []
    for key, value in updates.items():
        if hasattr(obj, key):
            previous[key] = getattr(obj, key)
        else:
            missing.append(key)
        setattr(obj, key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(obj, key, value)
        for key in missing:
            try:
                delattr(obj, key)
            except AttributeError:
                pass


def module_parameter_groups(model: torch.nn.Module) -> Dict[str, List[Tuple[str, torch.nn.Parameter]]]:
    """Assign parameters to interpretable PKD pipeline stages."""
    groups: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {
        "metric_depth": [],
        "seed_objectness_graspness": [],
        "view": [],
        "local_cva": [],
        "pre_cdf_decoder": [],
        "cdf_head": [],
        "other": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lower = name.lower()
        if "cdf_head" in lower:
            group = "cdf_head"
        elif "depth" in lower or "dpt" in lower or "pose_depth" in lower:
            group = "metric_depth"
        elif "view" in lower and not any(token in lower for token in ("kview", "query")):
            group = "view"
        elif any(token in lower for token in ("kview", "query_transformer", "local_region", "roi", "group")):
            group = "local_cva"
        elif "decoder" in lower:
            group = "pre_cdf_decoder"
        elif any(token in lower for token in ("objectness", "graspness", "seed", "point_head")):
            group = "seed_objectness_graspness"
        else:
            group = "other"
        groups[group].append((name, parameter))
    return groups


def flatten_grads(grads: Sequence[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    tensors = [grad.detach().reshape(-1).float() for grad in grads if grad is not None]
    return torch.cat(tensors) if tensors else None


def grad_relation(
    supervised_loss: torch.Tensor,
    kd_loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> Dict[str, float]:
    params = [parameter for parameter in parameters if parameter.requires_grad]
    if not params:
        return {"cosine": float("nan"), "sup_norm": 0.0, "kd_norm": 0.0, "kd_to_sup": float("nan"), "numel": 0}
    sup = torch.autograd.grad(supervised_loss, params, retain_graph=True, allow_unused=True)
    kd = torch.autograd.grad(kd_loss, params, retain_graph=True, allow_unused=True)
    sup_flat = flatten_grads(sup)
    kd_flat = flatten_grads(kd)
    if sup_flat is None or kd_flat is None:
        return {
            "cosine": float("nan"),
            "sup_norm": 0.0 if sup_flat is None else float(sup_flat.norm().item()),
            "kd_norm": 0.0 if kd_flat is None else float(kd_flat.norm().item()),
            "kd_to_sup": float("nan"),
            "numel": int(sum(parameter.numel() for parameter in params)),
        }
    sup_norm = sup_flat.norm()
    kd_norm = kd_flat.norm()
    cosine = torch.dot(sup_flat, kd_flat) / (sup_norm * kd_norm + 1e-12)
    return {
        "cosine": float(cosine.item()),
        "sup_norm": float(sup_norm.item()),
        "kd_norm": float(kd_norm.item()),
        "kd_to_sup": float((kd_norm / (sup_norm + 1e-12)).item()),
        "numel": int(sum(parameter.numel() for parameter in params)),
    }
