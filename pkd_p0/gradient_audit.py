"""Module-wise supervised-vs-PKD parameter-gradient audit.

The audit is injected immediately before the ordinary ``loss.backward()`` in
``Trainer.train_one_epoch``.  It uses two ``torch.autograd.grad`` traversals
(one for the supervised loss and one for the actual weighted distillation
loss), aggregates their relation by named model stage, and never writes to
``parameter.grad``.

This implementation is intentionally single-process.  PyTorch DDP explicitly
does not support ``torch.autograd.grad`` for model parameters, so launch P0-C2
with one visible GPU / WORLD_SIZE=1.

Environment variables
---------------------
PKD_P0_GRAD_AUDIT_DIR       required to enable the audit
PKD_P0_GRAD_AUDIT_EVERY     default: 200 training iterations
PKD_P0_GRAD_AUDIT_MAX       default: 30 audited iterations
PKD_P0_SUP_LOSS_KEY         exact end_points key (optional)
PKD_P0_KD_LOSS_KEY          exact end_points key (optional)
PKD_P0_TOTAL_LOSS_KEY       exact end_points key (optional)
PKD_P0_GRAD_GROUPS          comma-separated groups; defaults to the six P0 groups
PKD_P0_GRAD_STRICT          1: raise on contract error; 0: warn and continue
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .common import atomic_json_dump, module_parameter_groups, to_jsonable


AUDIT_VERSION = "pkd_p0_gradient_audit_v1_1"
DEFAULT_GROUPS = (
    "metric_depth",
    "seed_objectness_graspness",
    "view",
    "local_cva",
    "pre_cdf_decoder",
    "cdf_head",
)

SUP_ALIASES = (
    "A: Supervised Loss",
    "loss/supervised_loss",
    "loss/sup_loss",
    "loss/task_loss",
    "loss/grasp_loss",
    "loss/cdf_loss",
    "loss/grasp_cdf_loss",
    "loss/supervised_cdf_loss",
    "supervised_loss",
    "task_loss",
    "cdf_loss",
)
KD_ALIASES = (
    "A: Distill Loss",
    "loss/kd_loss",
    "loss/kd_total",
    "loss/privileged_kd_loss",
    "loss/kd_cdf_loss",
    "loss/privileged_kd_cdf",
    "kd_loss",
    "kd_total",
    "kd_cdf_loss",
)
TOTAL_ALIASES = (
    "A: Overall Loss",
    "loss/overall_loss",
    "loss/total_loss",
    "overall_loss",
    "total_loss",
    "loss",
)

_STATE: Dict[str, Any] = {
    "calls": 0,
    "audited": 0,
    "initialized": False,
    "warned": set(),
}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _scalar_tensor(value: Any) -> bool:
    return torch.is_tensor(value) and value.requires_grad and value.numel() == 1


def _distributed_identity() -> Tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = int(torch.distributed.get_rank())
        world_size = int(torch.distributed.get_world_size())
    return rank, world_size


def _find_endpoint_mapping(local_vars: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    # The instrumented EconomicGrasp trainer has an explicit ``end_points``
    # local.  Prefer it over heuristic scoring: ``batch_data_label`` can contain
    # many tensors and was occasionally selected by the old implementation.
    for name in ("end_points", "outputs", "output", "model_output"):
        value = local_vars.get(name)
        if isinstance(value, Mapping):
            return value

    candidates: List[Tuple[int, Mapping[str, Any]]] = []
    for value in local_vars.values():
        if not isinstance(value, Mapping):
            continue
        score = 0
        for key, item in value.items():
            if not torch.is_tensor(item):
                continue
            score += 4 if "loss" in str(key).lower() and item.numel() == 1 else 1
        if score:
            candidates.append((score, value))
    return max(candidates, key=lambda pair: pair[0])[1] if candidates else None


def _unwrap_module(value: Any) -> Optional[torch.nn.Module]:
    if not isinstance(value, torch.nn.Module):
        return None
    module = value
    # DDP/DataParallel wrappers expose ``module``.  The audit is required to run
    # with world size 1, but unwrapping also keeps names stable under wrappers.
    while hasattr(module, "module") and isinstance(module.module, torch.nn.Module):
        module = module.module
    return module


def _find_model(local_vars: Mapping[str, Any]) -> Tuple[Optional[torch.nn.Module], str]:
    # The active train_one_epoch locals contain ``self`` (Trainer), not a direct
    # nn.Module variable.  Resolve the trainer first.
    trainer = local_vars.get("self")
    if trainer is not None:
        unwrap = getattr(trainer, "unwrap_model", None)
        if callable(unwrap):
            candidate = _unwrap_module(unwrap())
            if candidate is not None:
                return candidate, "self.unwrap_model()"
        for attr in ("net", "model", "student"):
            candidate = _unwrap_module(getattr(trainer, attr, None))
            if candidate is not None:
                return candidate, f"self.{attr}"

    # Fallback for other trainer implementations.
    preferred = ("model", "net", "student", "network")
    for name in preferred:
        candidate = _unwrap_module(local_vars.get(name))
        if candidate is not None:
            return candidate, name

    candidates: List[Tuple[int, str, torch.nn.Module]] = []
    for name, value in local_vars.items():
        candidate = _unwrap_module(value)
        if candidate is None:
            continue
        trainable = sum(
            parameter.numel()
            for parameter in candidate.parameters()
            if parameter.requires_grad
        )
        candidates.append((trainable, str(name), candidate))
    if not candidates:
        return None, ""
    _, name, model = max(candidates, key=lambda item: item[0])
    return model, name


def _find_loss(
    mapping: Mapping[str, Any],
    explicit: str,
    aliases: Sequence[str],
) -> Tuple[Optional[str], Optional[torch.Tensor]]:
    if explicit:
        value = mapping.get(explicit)
        if _scalar_tensor(value):
            return explicit, value
        return None, None

    for key in aliases:
        value = mapping.get(key)
        if _scalar_tensor(value):
            return key, value

    normalized = {_normalize(key): key for key in mapping}
    for alias in aliases:
        key = normalized.get(_normalize(alias))
        if key is not None and _scalar_tensor(mapping[key]):
            return str(key), mapping[key]
    return None, None


def _loss_from_local(
    local_vars: Mapping[str, Any],
    names: Sequence[str],
) -> Tuple[Optional[str], Optional[torch.Tensor]]:
    for name in names:
        value = local_vars.get(name)
        if _scalar_tensor(value):
            return f"local:{name}", value
    return None, None


def _resolve_losses(
    local_vars: Mapping[str, Any],
    mapping: Mapping[str, Any],
) -> Tuple[
    Optional[str], Optional[torch.Tensor],
    Optional[str], Optional[torch.Tensor],
    Optional[str], Optional[torch.Tensor],
]:
    sup_key, sup_loss = _find_loss(
        mapping,
        os.environ.get("PKD_P0_SUP_LOSS_KEY", "").strip(),
        SUP_ALIASES,
    )
    kd_key, kd_loss = _find_loss(
        mapping,
        os.environ.get("PKD_P0_KD_LOSS_KEY", "").strip(),
        KD_ALIASES,
    )
    total_key, total_loss = _find_loss(
        mapping,
        os.environ.get("PKD_P0_TOTAL_LOSS_KEY", "").strip(),
        TOTAL_ALIASES,
    )

    # Known locals of train_cva_distill_ddp.py.  This fallback also proves that
    # the audit uses the exact tensors that are about to participate in
    # ``loss.backward()``.
    if sup_loss is None:
        sup_key, sup_loss = _loss_from_local(
            local_vars, ("supervised_loss", "task_loss")
        )
    if kd_loss is None:
        kd_key, kd_loss = _loss_from_local(
            local_vars, ("distill_loss", "kd_loss")
        )
    if total_loss is None:
        total_key, total_loss = _loss_from_local(
            local_vars, ("loss", "overall_loss", "total_loss")
        )
    return sup_key, sup_loss, kd_key, kd_loss, total_key, total_loss


def _find_step(local_vars: Mapping[str, Any]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    aliases = {
        "epoch": ("epoch", "epoch_idx", "epoch_index"),
        "batch": ("batch_idx", "batch_id", "batch_index", "it", "iteration"),
        "global_step": (
            "optimizer_step",
            "global_step",
            "step",
            "iter_num",
            "iteration_count",
        ),
    }
    for canonical, names in aliases.items():
        for name in names:
            value = local_vars.get(name)
            if isinstance(value, (int, bool)):
                result[canonical] = int(value)
                break
            if torch.is_tensor(value) and value.numel() == 1:
                result[canonical] = int(value.detach().item())
                break
    return result


def _scalar_diagnostics(mapping: Mapping[str, Any]) -> Dict[str, float]:
    wanted = (
        "teacher_better",
        "common_valid",
        "center_z",
        "support_iou",
        "kd_valid",
        "teacher_entropy",
        "student_entropy",
        "depth_mae",
    )
    result: Dict[str, float] = {}
    for key, value in mapping.items():
        lower = str(key).lower()
        if not any(token in lower for token in wanted):
            continue
        if torch.is_tensor(value) and value.numel() == 1:
            result[str(key)] = float(value.detach().item())
        elif isinstance(value, (int, float)):
            result[str(key)] = float(value)
    return result


def _available_loss_keys(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        str(key): {
            "shape": list(value.shape),
            "requires_grad": bool(value.requires_grad),
            "value": float(value.detach().item()) if value.numel() == 1 else None,
        }
        for key, value in mapping.items()
        if torch.is_tensor(value) and "loss" in str(key).lower()
    }


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")


def _requested_groups(
    all_groups: Mapping[str, Sequence[Tuple[str, torch.nn.Parameter]]],
) -> Tuple[str, ...]:
    text = os.environ.get("PKD_P0_GRAD_GROUPS", "").strip()
    requested = tuple(
        token.strip() for token in text.split(",") if token.strip()
    ) if text else DEFAULT_GROUPS
    unknown = sorted(set(requested) - set(all_groups))
    if unknown:
        raise RuntimeError(
            f"Unknown PKD_P0_GRAD_GROUPS={unknown}; available={sorted(all_groups)}"
        )
    return requested


def _relation_from_aligned_grads(
    named_parameters: Sequence[Tuple[str, torch.nn.Parameter]],
    gradient_by_id_sup: Mapping[int, Optional[torch.Tensor]],
    gradient_by_id_kd: Mapping[int, Optional[torch.Tensor]],
) -> Dict[str, float]:
    dot = torch.zeros((), device=next(iter(named_parameters))[1].device, dtype=torch.float64)
    sup_sq = torch.zeros_like(dot)
    kd_sq = torch.zeros_like(dot)
    numel = 0
    sup_connected = 0
    kd_connected = 0
    shared_connected = 0

    for _, parameter in named_parameters:
        numel += int(parameter.numel())
        sup_grad = gradient_by_id_sup.get(id(parameter))
        kd_grad = gradient_by_id_kd.get(id(parameter))
        if sup_grad is not None:
            sup_connected += int(parameter.numel())
            sup_value = sup_grad.detach().double()
            sup_sq = sup_sq + torch.sum(sup_value * sup_value)
        else:
            sup_value = None
        if kd_grad is not None:
            kd_connected += int(parameter.numel())
            kd_value = kd_grad.detach().double()
            kd_sq = kd_sq + torch.sum(kd_value * kd_value)
        else:
            kd_value = None
        if sup_value is not None and kd_value is not None:
            shared_connected += int(parameter.numel())
            dot = dot + torch.sum(sup_value * kd_value)

    sup_norm = torch.sqrt(sup_sq)
    kd_norm = torch.sqrt(kd_sq)
    denominator = sup_norm * kd_norm
    cosine = (
        float((dot / denominator).item())
        if float(denominator.item()) > 0.0
        else float("nan")
    )
    ratio = (
        float((kd_norm / sup_norm).item())
        if float(sup_norm.item()) > 0.0
        else float("nan")
    )
    return {
        "cosine": cosine,
        "sup_norm": float(sup_norm.item()),
        "kd_norm": float(kd_norm.item()),
        "kd_to_sup": ratio,
        "numel": int(numel),
        "sup_connected_numel": int(sup_connected),
        "kd_connected_numel": int(kd_connected),
        "shared_connected_numel": int(shared_connected),
    }


def _all_group_relations(
    supervised_loss: torch.Tensor,
    kd_loss: torch.Tensor,
    groups: Mapping[str, Sequence[Tuple[str, torch.nn.Parameter]]],
) -> Dict[str, Dict[str, float]]:
    # Build one unique ordered parameter list.  The old implementation invoked
    # autograd twice per group; this version requires only two graph traversals
    # in total and preserves exact parameter alignment when one loss does not
    # connect to some parameters.
    parameters: List[torch.nn.Parameter] = []
    seen = set()
    for named_parameters in groups.values():
        for _, parameter in named_parameters:
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            parameters.append(parameter)

    if not parameters:
        return {
            group: {
                "cosine": float("nan"),
                "sup_norm": 0.0,
                "kd_norm": 0.0,
                "kd_to_sup": float("nan"),
                "numel": 0,
                "sup_connected_numel": 0,
                "kd_connected_numel": 0,
                "shared_connected_numel": 0,
            }
            for group in groups
        }

    sup_grads = torch.autograd.grad(
        supervised_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    kd_grads = torch.autograd.grad(
        kd_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    sup_by_id = {id(parameter): grad for parameter, grad in zip(parameters, sup_grads)}
    kd_by_id = {id(parameter): grad for parameter, grad in zip(parameters, kd_grads)}

    relations = {
        group: _relation_from_aligned_grads(named_parameters, sup_by_id, kd_by_id)
        for group, named_parameters in groups.items()
        if named_parameters
    }
    del sup_grads, kd_grads, sup_by_id, kd_by_id
    return relations


def _warn_once(message: str) -> None:
    warned = _STATE["warned"]
    if message not in warned:
        print(f"[PKD-P0-GRAD][WARN] {message}", flush=True)
        warned.add(message)


def maybe_audit_from_training_locals(local_vars: Mapping[str, Any]) -> None:
    output_text = os.environ.get("PKD_P0_GRAD_AUDIT_DIR", "").strip()
    if not output_text:
        return

    rank, world_size = _distributed_identity()
    if world_size != 1:
        raise RuntimeError(
            "P0-C2 parameter-gradient audit requires WORLD_SIZE=1.  PyTorch "
            "DistributedDataParallel does not support torch.autograd.grad() "
            "for model parameters.  Re-run the same Stage-2 script with "
            "GPU_IDS=0 (one process); do not use nproc_per_node>1."
        )

    _STATE["calls"] += 1
    every = max(1, int(os.environ.get("PKD_P0_GRAD_AUDIT_EVERY", "200")))
    maximum = max(1, int(os.environ.get("PKD_P0_GRAD_AUDIT_MAX", "30")))
    if _STATE["audited"] >= maximum or (_STATE["calls"] - 1) % every != 0:
        return

    output_dir = Path(output_text).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    strict = os.environ.get("PKD_P0_GRAD_STRICT", "1") == "1"

    mapping = _find_endpoint_mapping(local_vars)
    model, model_source = _find_model(local_vars)
    if mapping is None or model is None:
        message = (
            "Could not identify end_points/model from training locals; "
            f"mapping_found={mapping is not None}, model_found={model is not None}, "
            f"names={sorted(local_vars)}"
        )
        if strict:
            raise RuntimeError(message)
        _warn_once(message)
        return

    (
        sup_key,
        sup_loss,
        kd_key,
        kd_loss,
        total_key,
        total_loss,
    ) = _resolve_losses(local_vars, mapping)

    if sup_loss is None and total_loss is not None and kd_loss is not None:
        weight_text = os.environ.get("PKD_P0_KD_WEIGHT", "").strip()
        if weight_text:
            sup_loss = total_loss - float(weight_text) * kd_loss
            sup_key = f"{total_key} - {weight_text}*{kd_key}"

    all_groups = module_parameter_groups(model)
    requested_names = _requested_groups(all_groups)
    groups = {
        name: all_groups[name]
        for name in requested_names
        if len(all_groups[name]) > 0
    }

    if not _STATE["initialized"]:
        atomic_json_dump(
            {
                "audit_version": AUDIT_VERSION,
                "rank": rank,
                "world_size": world_size,
                "model_source": model_source,
                "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
                "available_loss_keys": _available_loss_keys(mapping),
                "resolved": {
                    "supervised": sup_key,
                    "kd": kd_key,
                    "total": total_key,
                },
                "requested_parameter_groups": list(requested_names),
                "parameter_groups": {
                    group: [name for name, _ in values]
                    for group, values in groups.items()
                },
                "excluded_parameter_groups": {
                    group: len(values)
                    for group, values in all_groups.items()
                    if group not in groups
                },
                "environment": {
                    key: value
                    for key, value in os.environ.items()
                    if key.startswith("PKD_P0_")
                },
            },
            output_dir / "gradient_audit_contract.json",
        )
        _STATE["initialized"] = True

    if sup_loss is None or kd_loss is None:
        message = (
            "Could not resolve differentiable supervised/KD scalar losses. "
            f"resolved supervised={sup_key}, kd={kd_key}; available loss "
            f"keys={sorted(_available_loss_keys(mapping))}."
        )
        if strict:
            raise RuntimeError(message)
        _warn_once(message)
        return
    if not groups:
        raise RuntimeError(
            "No trainable parameters were found in the requested P0 gradient groups."
        )

    relations = _all_group_relations(sup_loss, kd_loss, groups)
    step = _find_step(local_vars)
    payload = {
        "audit_version": AUDIT_VERSION,
        "audit_index": int(_STATE["audited"]),
        "call_index": int(_STATE["calls"]),
        "rank": rank,
        "world_size": world_size,
        "time": time.time(),
        **step,
        "supervised_loss_key": sup_key,
        "kd_loss_key": kd_key,
        "total_loss_key": total_key,
        "supervised_loss": float(sup_loss.detach().item()),
        "kd_loss": float(kd_loss.detach().item()),
        "total_loss": (
            float(total_loss.detach().item())
            if total_loss is not None
            else None
        ),
        "gradient_relations": relations,
        "batch_diagnostics": _scalar_diagnostics(mapping),
    }
    _append_jsonl(output_dir / "gradient_audit.jsonl", payload)
    _STATE["audited"] += 1

    summary = ", ".join(
        f"{group}:cos={values['cosine']:.3f},ratio={values['kd_to_sup']:.2f}"
        for group, values in relations.items()
        if math.isfinite(values["cosine"])
    )
    print(
        f"[PKD-P0-GRAD] {_STATE['audited']}/{maximum} "
        f"step={step.get('global_step', _STATE['calls'])} {summary}",
        flush=True,
    )
