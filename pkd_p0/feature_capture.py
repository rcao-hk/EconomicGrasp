"""Named feature capture for the current EconomicGrasp CVA-CDF pipeline.

The current CenterViewAngle module invokes most submodules with keyword-only
arguments.  PyTorch's default forward-pre-hook receives positional ``args``
only, which is an empty tuple for calls such as::

    self.kview_grasp_module(seed_features=..., seed_xyz=..., ...)

This module registers keyword-aware hooks and captures semantically defined
features from the current pipeline without modifying repository model files.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch


# Public compatibility constant.  The optional third tuple item names the
# semantic extractor; user-provided historical (pattern, mode) pairs continue
# to work through the generic first-tensor fallback.
FEATURE_PATTERNS = {
    # Image-FPS seed representation entering the complete CVA module.
    "seed": (
        r"(?:^|\.)kview_grasp_module$",
        "input",
        "seed_features",
    ),
    # Feature after the ViewNet residual update: seed_features + res_feat.
    # The same ViewNet instance is registered as both model.view and
    # model.kview_grasp_module.view, so named_modules(remove_duplicate=False)
    # is used below and the longest fully-qualified alias is preferred.
    "pre_view": (
        r"(?:^|\.)(?:kview_grasp_module\.view|view)$",
        "view_fused_output",
        "seed_features",
    ),
    # Center-view query feature returned by KViewQuerySelector.
    "selected_view": (
        r"(?:^|\.)kview_grasp_module\.selector$",
        "output",
        "output_0",
    ),
    # Angle-conditioned local representation returned by the grouping module.
    "local": (
        r"(?:^|\.)kview_grasp_module\.group$",
        "output",
        "output_tensor",
    ),
    # Exact tensor consumed by the current final CDF Conv1d.
    "pre_cdf": (
        r"(?:^|\.)kview_grasp_module\.decoder\.cdf_head$",
        "input",
        "input_tensor",
    ),
}


def first_tensor(value: Any) -> Optional[torch.Tensor]:
    """Depth-first tensor lookup retained for custom generic capture specs."""
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _argument_tensor(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    key: Optional[str],
) -> Optional[torch.Tensor]:
    if key and key in kwargs and torch.is_tensor(kwargs[key]):
        return kwargs[key]
    tensor = first_tensor(args)
    if tensor is not None:
        return tensor
    return first_tensor(kwargs)


def _output_index_tensor(output: Any, index: int) -> Optional[torch.Tensor]:
    if isinstance(output, (tuple, list)) and len(output) > index:
        value = output[index]
        if torch.is_tensor(value):
            return value
    return None


@dataclass(frozen=True)
class CaptureSpec:
    pattern: str
    mode: str
    selector: Optional[str] = None


@dataclass
class Match:
    layer: str
    module_name: str
    mode: str
    selector: Optional[str] = None


def _normalize_spec(value: Union[CaptureSpec, Sequence[str]]) -> CaptureSpec:
    if isinstance(value, CaptureSpec):
        return value
    if not isinstance(value, (tuple, list)) or len(value) not in (2, 3):
        raise TypeError(
            "Feature capture spec must be CaptureSpec, (pattern, mode), or "
            f"(pattern, mode, selector); got {value!r}."
        )
    pattern = str(value[0])
    mode = str(value[1])
    selector = None if len(value) == 2 else str(value[2])
    return CaptureSpec(pattern=pattern, mode=mode, selector=selector)


class FeatureCapture:
    """Capture current CVA representations using non-invasive runtime hooks."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        patterns: Optional[Mapping[str, Union[CaptureSpec, Sequence[str]]]] = None,
        strict: bool = False,
    ) -> None:
        self.model = model
        raw_patterns = patterns or FEATURE_PATTERNS
        self.patterns = {
            str(layer): _normalize_spec(spec)
            for layer, spec in raw_patterns.items()
        }
        self.strict = bool(strict)
        self.handles: List[Any] = []
        self.matches: List[Match] = []
        self.values: Dict[str, torch.Tensor] = {}
        self.capture_errors: Dict[str, str] = {}

        # ViewNet is shared by the outer model and kview_grasp_module.  Keeping
        # duplicate aliases makes the intended nested path discoverable.
        try:
            named = list(model.named_modules(remove_duplicate=False))
        except TypeError:  # older PyTorch compatibility
            named = list(model.named_modules())

        for layer, spec in self.patterns.items():
            regex = re.compile(spec.pattern, re.IGNORECASE)
            candidates = [
                (name, module)
                for name, module in named
                if regex.search(name)
            ]
            if len(candidates) > 1:
                # Prefer the most explicit nested alias.  If multiple aliases
                # have the same length but reference the same module object,
                # retain one deterministically.
                max_length = max(len(name) for name, _ in candidates)
                candidates = [
                    (name, module)
                    for name, module in candidates
                    if len(name) == max_length
                ]
                unique_by_id: Dict[int, Tuple[str, torch.nn.Module]] = {}
                for name, module in candidates:
                    unique_by_id.setdefault(id(module), (name, module))
                candidates = list(unique_by_id.values())

            if not candidates:
                message = (
                    f"No module matched feature layer {layer!r}: "
                    f"{spec.pattern}"
                )
                if self.strict:
                    raise RuntimeError(message)
                self.capture_errors[layer] = message
                continue
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Ambiguous module pattern for {layer}: "
                    f"{[name for name, _ in candidates]}"
                )

            name, module = candidates[0]
            self.matches.append(
                Match(layer, name, spec.mode, spec.selector)
            )
            if spec.mode == "input":
                handle = module.register_forward_pre_hook(
                    self._make_pre_hook(layer, spec.selector),
                    with_kwargs=True,
                )
            elif spec.mode == "output":
                handle = module.register_forward_hook(
                    self._make_post_hook(layer, spec.selector),
                    with_kwargs=True,
                )
            elif spec.mode == "view_fused_output":
                handle = module.register_forward_hook(
                    self._make_view_fused_hook(layer, spec.selector),
                    with_kwargs=True,
                )
            else:
                raise ValueError(f"Unknown capture mode {spec.mode!r}")
            self.handles.append(handle)

    def _store_or_report(
        self,
        layer: str,
        tensor: Optional[torch.Tensor],
        message: str,
    ) -> None:
        if tensor is None:
            self.capture_errors[layer] = message
            if self.strict:
                raise RuntimeError(message)
            return
        if not torch.is_tensor(tensor):
            message = (
                f"Captured value for layer {layer!r} is not a tensor: "
                f"{type(tensor).__name__}."
            )
            self.capture_errors[layer] = message
            if self.strict:
                raise RuntimeError(message)
            return
        self.values[layer] = tensor.detach()
        self.capture_errors.pop(layer, None)

    def _make_pre_hook(self, layer: str, selector: Optional[str]):
        def hook(
            _module: torch.nn.Module,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> None:
            key = selector if selector not in {None, "input_tensor"} else None
            tensor = _argument_tensor(args, kwargs, key)
            self._store_or_report(
                layer,
                tensor,
                (
                    f"No tensor in positional or keyword inputs for captured "
                    f"layer {layer!r}; args={len(args)}, "
                    f"kwargs={sorted(kwargs)}."
                ),
            )
        return hook

    def _make_post_hook(self, layer: str, selector: Optional[str]):
        def hook(
            _module: torch.nn.Module,
            _args: Tuple[Any, ...],
            _kwargs: Dict[str, Any],
            output: Any,
        ) -> None:
            tensor: Optional[torch.Tensor]
            index_match = (
                re.fullmatch(r"output_(\d+)", selector)
                if selector is not None
                else None
            )
            if index_match is not None:
                tensor = _output_index_tensor(
                    output,
                    int(index_match.group(1)),
                )
            else:
                tensor = first_tensor(output)
            self._store_or_report(
                layer,
                tensor,
                f"No tensor in output for captured layer {layer!r}.",
            )
        return hook

    def _make_view_fused_hook(
        self,
        layer: str,
        selector: Optional[str],
    ):
        """Capture seed_features + ViewNet residual, the selector input."""
        def hook(
            _module: torch.nn.Module,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
            output: Any,
        ) -> None:
            seed = _argument_tensor(args, kwargs, selector or "seed_features")
            residual = _output_index_tensor(output, 1)
            tensor: Optional[torch.Tensor] = None
            if seed is not None and residual is not None:
                if tuple(seed.shape) != tuple(residual.shape):
                    message = (
                        f"ViewNet seed/residual shape mismatch for {layer!r}: "
                        f"seed={tuple(seed.shape)}, residual={tuple(residual.shape)}."
                    )
                    self.capture_errors[layer] = message
                    if self.strict:
                        raise RuntimeError(message)
                    return
                tensor = seed + residual
            self._store_or_report(
                layer,
                tensor,
                (
                    f"Could not capture fused pre-selector ViewNet feature for "
                    f"{layer!r}; seed={'yes' if seed is not None else 'no'}, "
                    f"residual={'yes' if residual is not None else 'no'}."
                ),
            )
        return hook

    def clear(self) -> None:
        self.values.clear()
        # Keep static module-discovery errors, but clear transient forward errors.
        matched_layers = {match.layer for match in self.matches}
        self.capture_errors = {
            layer: message
            for layer, message in self.capture_errors.items()
            if layer not in matched_layers
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def contract(self) -> Dict[str, Any]:
        return {
            "matches": [match.__dict__ for match in self.matches],
            "captured_layers": sorted(self.values),
            "captured_shapes": {
                key: list(value.shape)
                for key, value in self.values.items()
            },
            "capture_errors": dict(sorted(self.capture_errors.items())),
            "keyword_aware_hooks": True,
            "feature_semantics": {
                "seed": "image-FPS seed feature entering CVA/ViewNet",
                "pre_view": "seed feature plus ViewNet residual, before selector",
                "selected_view": "selector output seed_features_q",
                "local": "angle-conditioned grouping output",
                "pre_cdf": "input to current decoder.cdf_head",
            },
        }
