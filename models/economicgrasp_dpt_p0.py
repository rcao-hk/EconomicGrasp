"""P0-only EconomicGrasp-DPT model wrappers.

This module adds the counterfactual controls required by the PKD P0
experiments without editing ``economicgrasp_bip3d.py`` or
``kview_query_transformer.py``.  It subclasses the current Stage-0 teacher and
Stage-1/2 student, adds no parameters or buffers, and therefore preserves the
checkpoint state-dict contract.

Runtime-only input keys
-----------------------
``_pkd_p0_geometry_depth_override``
    Replace only the metric depth consumed by the existing spatial enhancer,
    image-FPS backprojection, ViewNet, and local CVA module for one forward
    call.  The RGB student still executes its normal depth/image backbone; only
    the metric-depth output is replaced.  The clean-geometry teacher similarly
    keeps its normal RGB features while its prepared geometry depth is replaced.

``_pkd_p0_forced_query``
    Replay the student's ordered image-FPS seed pixels and selected view
    indices.  Existing repository controls are reused:
    ``image_fps_seed_idx_override`` for centers,
    ``oracle_view_inds_override`` for Top-1, and the existing P0-B
    selector-instance override for deterministic Top-K.  The teacher does not
    copy the student's 3D centers: it backprojects the same image pixels with
    its own active geometry, which is the intended physical counterfactual.
"""
from __future__ import annotations

import contextlib
from types import MethodType
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional, Tuple

import torch

from .economicgrasp_dpt_distill import (
    DISTILL_CONTRACT_VERSION,
    OutputDistillationConfig,
    compute_output_distillation_loss,
    economicgrasp_dpt_student as _CurrentStudent,
    economicgrasp_dpt_teacher as _CurrentTeacher,
    extract_distillation_targets,
    load_checkpoint_state,
)
from .p0b_topk_exact_override import (
    P0B_EXACT_QUERY_VIEW_OVERRIDE_KEY,
    install_p0b_exact_query_selector_override,
)


P0_RUNTIME_SCHEMA_VERSION = 2
P0_GEOMETRY_OVERRIDE_KEY = "_pkd_p0_geometry_depth_override"
P0_FORCED_QUERY_KEY = "_pkd_p0_forced_query"
P0_GEOMETRY_MARKER = "D: P0 geometry override active"
P0_QUERY_MARKER = "D: P0 query override active"
# Compatibility aliases used by the first overlay revision.
P0_GEOMETRY_MARKER_KEY = P0_GEOMETRY_MARKER
P0_QUERY_MARKER_KEY = P0_QUERY_MARKER


class P0RuntimeContractError(RuntimeError):
    """Raised when a P0 counterfactual is not applied exactly."""


def runtime_contract() -> Dict[str, Any]:
    """Return the static contract of the source-free P0 runtime."""
    return {
        "schema_version": P0_RUNTIME_SCHEMA_VERSION,
        "source_patch_required": False,
        "geometry_override_key": P0_GEOMETRY_OVERRIDE_KEY,
        "forced_query_key": P0_FORCED_QUERY_KEY,
        "geometry_marker": P0_GEOMETRY_MARKER,
        "query_marker": P0_QUERY_MARKER,
        "query_alignment": (
            "same ordered image-FPS pixels and selected view indices; "
            "teacher XYZ is recomputed from the teacher's active geometry"
        ),
        "state_dict_changed": False,
        "base_source_files_modified": [],
        "topk_eval_replay": True,
        "topk_training_replay": False,
        "single_view_training_rank_normalization": True,
        "accepted_single_view_source_ranks": [-2, -1, 0],
    }


def _scalar_like(reference: torch.Tensor, value: float) -> torch.Tensor:
    return reference.new_tensor(float(value)).reshape(())


def _normalize_depth_override(
    value: torch.Tensor,
    *,
    batch_size: int,
    image_hw: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(
            f"{P0_GEOMETRY_OVERRIDE_KEY} must be a torch.Tensor, got "
            f"{type(value)!r}."
        )
    depth = value.detach()
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    elif depth.dim() == 4:
        if depth.shape[1] != 1:
            raise ValueError(
                f"{P0_GEOMETRY_OVERRIDE_KEY} must have one depth channel; "
                f"got {tuple(depth.shape)}."
            )
    else:
        raise ValueError(
            f"{P0_GEOMETRY_OVERRIDE_KEY} must be [B,H,W] or [B,1,H,W], "
            f"got {tuple(depth.shape)}."
        )
    expected = (int(batch_size), 1, int(image_hw[0]), int(image_hw[1]))
    if not bool(torch.isfinite(depth).all()):
        raise ValueError(
            f"{P0_GEOMETRY_OVERRIDE_KEY} contains NaN or Inf."
        )
    if tuple(depth.shape) != expected:
        raise ValueError(
            f"{P0_GEOMETRY_OVERRIDE_KEY} shape mismatch: expected {expected}, "
            f"got {tuple(depth.shape)}. P0 deliberately does not resize a "
            "counterfactual geometry tensor."
        )
    return depth.to(device=device, dtype=dtype).contiguous()


def _replace_depth_net_output(
    result: Any,
    override: torch.Tensor,
    *,
    batch_size: int,
    image_hw: Tuple[int, int],
) -> Any:
    """Replace the current metric-depth output while preserving all features."""
    if torch.is_tensor(result):
        return _normalize_depth_override(
            override,
            batch_size=batch_size,
            image_hw=image_hw,
            device=result.device,
            dtype=result.dtype,
        )

    if isinstance(result, tuple):
        if not result or not torch.is_tensor(result[0]):
            raise P0RuntimeContractError(
                "Current depth_net tuple output no longer starts with the "
                "metric-depth tensor."
            )
        replacement = _normalize_depth_override(
            override,
            batch_size=batch_size,
            image_hw=image_hw,
            device=result[0].device,
            dtype=result[0].dtype,
        )
        return (replacement, *result[1:])

    if isinstance(result, list):
        if not result or not torch.is_tensor(result[0]):
            raise P0RuntimeContractError(
                "Current depth_net list output no longer starts with the "
                "metric-depth tensor."
            )
        copied = list(result)
        copied[0] = _normalize_depth_override(
            override,
            batch_size=batch_size,
            image_hw=image_hw,
            device=result[0].device,
            dtype=result[0].dtype,
        )
        return copied

    if isinstance(result, MutableMapping):
        aliases = (
            "depth",
            "depth_pred",
            "metric_depth",
            "metric_depth_pred",
            "depth_448",
        )
        keys = [key for key in aliases if torch.is_tensor(result.get(key))]
        if len(keys) != 1:
            raise P0RuntimeContractError(
                "Could not uniquely identify the metric-depth tensor in the "
                f"depth_net mapping output; candidates={keys}."
            )
        key = keys[0]
        reference = result[key]
        copied = dict(result)
        copied[key] = _normalize_depth_override(
            override,
            batch_size=batch_size,
            image_hw=image_hw,
            device=reference.device,
            dtype=reference.dtype,
        )
        return copied

    raise P0RuntimeContractError(
        "Unsupported current depth_net output type for P0 geometry override: "
        f"{type(result)!r}."
    )


@contextlib.contextmanager
def _temporary_instance_method(
    obj: Any,
    name: str,
    function: Any,
) -> Iterator[None]:
    """Temporarily bind one method on one instance and restore it exactly."""
    had_instance_value = name in getattr(obj, "__dict__", {})
    old_instance_value = obj.__dict__.get(name) if had_instance_value else None
    setattr(obj, name, MethodType(function, obj))
    try:
        yield
    finally:
        if had_instance_value:
            setattr(obj, name, old_instance_value)
        else:
            try:
                delattr(obj, name)
            except AttributeError:
                pass


def _require_rank2_long(
    value: Any,
    *,
    name: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"P0 forced query requires tensor {name!r}.")
    tensor = value.detach()
    if device is not None:
        tensor = tensor.to(device=device)
    tensor = tensor.to(dtype=torch.long)
    if tensor.dim() != 2:
        raise ValueError(
            f"P0 forced-query tensor {name!r} must be rank 2, got "
            f"{tuple(tensor.shape)}."
        )
    return tensor.contiguous()


def _canonicalize_query_contract(
    query: Mapping[str, Any],
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if not isinstance(query, Mapping):
        raise TypeError(f"P0 query contract must be a mapping, got {type(query)!r}.")

    base_value = query.get("kview_base_token_sel_idx")
    if base_value is None:
        base_value = query.get("image_fps_seed_idx_override")
    base = _require_rank2_long(
        base_value,
        name="kview_base_token_sel_idx",
        device=device,
    )
    token = _require_rank2_long(
        query.get("token_sel_idx"),
        name="token_sel_idx",
        device=device,
    )
    view = _require_rank2_long(
        query.get("grasp_top_view_inds"),
        name="grasp_top_view_inds",
        device=device,
    )

    B, M = base.shape
    if token.shape != view.shape or token.shape[0] != B:
        raise ValueError(
            "P0 forced-query token/view shapes differ: "
            f"token={tuple(token.shape)}, view={tuple(view.shape)}, B={B}."
        )
    Q = int(token.shape[1])
    if M <= 0 or Q <= 0 or Q % M != 0:
        raise ValueError(
            f"P0 forced-query Q={Q} must be a positive multiple of M={M}."
        )
    K = Q // M

    expected_token = (
        base.unsqueeze(-1).expand(B, M, K).reshape(B, Q).contiguous()
    )
    if not torch.equal(token, expected_token):
        raise P0RuntimeContractError(
            "P0 requires center-major image queries: token_sel_idx must repeat "
            "each ordered image-FPS seed K times."
        )

    result: Dict[str, Any] = {
        "kview_base_token_sel_idx": base,
        "token_sel_idx": token,
        "grasp_top_view_inds": view,
        "kview_effective_k_int": int(K),
    }

    parent = query.get("kview_query_parent")
    if parent is not None:
        parent_t = _require_rank2_long(
            parent,
            name="kview_query_parent",
            device=device,
        )
        expected_parent = (
            torch.arange(M, device=base.device, dtype=torch.long)
            .view(1, M, 1)
            .expand(B, M, K)
            .reshape(B, Q)
            .contiguous()
        )
        if parent_t.shape != expected_parent.shape or not torch.equal(
            parent_t, expected_parent
        ):
            raise P0RuntimeContractError(
                "P0 query contract has non-center-major kview_query_parent."
            )
        result["kview_query_parent"] = parent_t

    # ``kview_query_view_rank`` has two meanings in the current selector:
    # deterministic evaluation uses ordinal ranks 0..K-1, while training uses
    # provenance sentinels (-2 for the ViewNet sample reused by A1/A2 and -1
    # for a selector-sampled view).  P0-C deliberately keeps the student's
    # constructor-time training policy so its query distribution matches PKD
    # training.  For K=1, either sentinel still denotes the sole structural
    # query slot, whose canonical rank is unambiguously zero.
    rank = query.get("kview_query_view_rank")
    if rank is not None:
        rank_t = _require_rank2_long(
            rank,
            name="kview_query_view_rank",
            device=device,
        )
        expected_rank = (
            torch.arange(K, device=base.device, dtype=torch.long)
            .view(1, 1, K)
            .expand(B, M, K)
            .reshape(B, Q)
            .contiguous()
        )
        if rank_t.shape != expected_rank.shape:
            raise P0RuntimeContractError(
                "P0 query view-rank shape mismatch: "
                f"actual={tuple(rank_t.shape)}, "
                f"expected={tuple(expected_rank.shape)}."
            )

        if torch.equal(rank_t, expected_rank):
            canonical_rank = rank_t
        elif K == 1:
            # A single-view training query may carry either current training
            # sentinel.  Mixed 0/-1/-2 is also structurally valid because each
            # center still has exactly one explicitly stored selected view.
            allowed = (rank_t == 0) | (rank_t == -1) | (rank_t == -2)
            if not bool(allowed.all()):
                bad = sorted(
                    int(value)
                    for value in torch.unique(rank_t[~allowed])
                    .detach()
                    .cpu()
                    .tolist()
                )
                raise P0RuntimeContractError(
                    "P0 single-view query ranks must be 0, -1, or -2; "
                    f"unsupported values={bad}."
                )
            # Normalize only the metadata rank.  Exact physical correspondence
            # remains enforced below by center-major token_sel_idx and the
            # explicit grasp_top_view_inds tensor.
            canonical_rank = expected_rank
            result["p0_source_kview_query_view_rank"] = rank_t
        else:
            # Do not silently canonicalize sampled Top-K training queries.  For
            # K>1, -1/-2 do not identify a stable ordinal ordering and the P0
            # runtime intentionally supports deterministic evaluation replay
            # only.
            unique = sorted(
                int(value)
                for value in torch.unique(rank_t).detach().cpu().tolist()
            )
            raise P0RuntimeContractError(
                "P0 Top-K query replay requires deterministic view ranks "
                f"0..{K - 1}; received values={unique}."
            )

        result["kview_query_view_rank"] = canonical_rank

    if K > 1:
        views_bmk = view.view(B, M, K)
        sorted_views = torch.sort(views_bmk, dim=-1).values
        if bool((sorted_views[..., 1:] == sorted_views[..., :-1]).any()):
            raise P0RuntimeContractError(
                "P0 Top-K query contract contains duplicate views within one center."
            )
    return result


def extract_p0_query_contract(output: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the exact image-pixel/view contract from a student output.

    Deliberately excluded are ``xyz_graspable`` and ``grasp_top_view_xyz``.
    Those physical quantities must be recomputed by the teacher from its own
    active geometry after the image-space query has been fixed.
    """
    contract = _canonicalize_query_contract(output)
    return {
        key: (value.detach().clone() if torch.is_tensor(value) else value)
        for key, value in contract.items()
    }


def assert_p0_exact_query_output(
    expected_query: Mapping[str, Any],
    output: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    """Fail unless an output exactly reuses the requested pixels and views."""
    expected = _canonicalize_query_contract(expected_query)
    actual = _canonicalize_query_contract(output)

    reference = actual["token_sel_idx"]
    diagnostics: Dict[str, torch.Tensor] = {}
    for key, label in (
        ("kview_base_token_sel_idx", "base seed pixel"),
        ("token_sel_idx", "expanded query pixel"),
        ("grasp_top_view_inds", "selected view"),
    ):
        lhs = actual[key]
        rhs = expected[key].to(device=lhs.device, dtype=lhs.dtype)
        if lhs.shape != rhs.shape or not torch.equal(lhs, rhs):
            mismatch = 1.0
            if lhs.shape == rhs.shape and lhs.numel() > 0:
                mismatch = float((lhs != rhs).float().mean().item())
            raise P0RuntimeContractError(
                f"P0 exact {label} mismatch: actual={tuple(lhs.shape)}, "
                f"expected={tuple(rhs.shape)}, mismatch={mismatch:.6f}."
            )
        diagnostics[f"D: P0 exact {label} ratio"] = _scalar_like(lhs, 1.0)

    if int(actual["kview_effective_k_int"]) != int(
        expected["kview_effective_k_int"]
    ):
        raise P0RuntimeContractError(
            "P0 exact query effective-K mismatch: "
            f"actual={actual['kview_effective_k_int']}, "
            f"expected={expected['kview_effective_k_int']}."
        )

    for key, label in (
        ("kview_query_parent", "query-parent ordering"),
        ("kview_query_view_rank", "view-rank ordering"),
    ):
        if key not in expected:
            continue
        if key not in actual:
            raise P0RuntimeContractError(f"P0 output is missing {label} ({key}).")
        lhs = actual[key]
        rhs = expected[key].to(device=lhs.device, dtype=lhs.dtype)
        if lhs.shape != rhs.shape or not torch.equal(lhs, rhs):
            raise P0RuntimeContractError(f"P0 exact {label} mismatch.")

    diagnostics["D: P0 exact query K"] = _scalar_like(
        reference, float(actual["kview_effective_k_int"])
    )
    diagnostics["D: P0 query alignment is image-space"] = _scalar_like(
        reference, 1.0
    )
    return diagnostics


def build_p0_exact_query_input(
    batch: Mapping[str, Any],
    forced_query: Mapping[str, Any],
    *,
    force_process_grasp_labels: bool = True,
) -> Dict[str, Any]:
    """Build a one-forward input for exact image-pixel/view replay."""
    local = dict(batch)
    local[P0_FORCED_QUERY_KEY] = _canonicalize_query_contract(forced_query)
    local["cva_force_process_grasp_labels"] = bool(force_process_grasp_labels)
    local["cva_compute_diagnostics"] = False
    local["geometry_compute_diagnostics"] = False
    local["cva_export_angle_feature"] = False
    return local


def forward_with_p0_geometry_override(
    model: torch.nn.Module,
    end_points: Mapping[str, Any],
    geometry_override: Optional[torch.Tensor] = None,
) -> MutableMapping[str, Any]:
    """Run one model call with an optional source-free geometry intervention."""
    local: Dict[str, Any] = dict(end_points)
    if geometry_override is not None:
        local[P0_GEOMETRY_OVERRIDE_KEY] = geometry_override
    output = model(local)
    if not isinstance(output, MutableMapping):
        raise TypeError(
            "Current EconomicGrasp P0 model returned a non-mapping: "
            f"{type(output)!r}."
        )
    return output


def enable_p0_exact_query_runtime(model: torch.nn.Module) -> torch.nn.Module:
    """Compatibility no-op that validates use of a dedicated P0 constructor."""
    if not isinstance(model, _P0RuntimeMixin):
        raise TypeError(
            "P0 runtime controls require economicgrasp_dpt_p0_student or "
            "economicgrasp_dpt_p0_teacher; an unwrapped repository model was "
            "provided."
        )
    return model


class _P0RuntimeMixin:
    """Instance-local runtime controls shared by the P0 teacher and student."""

    def _p0_geometry_context(
        self,
        override: Optional[torch.Tensor],
        end_points: Mapping[str, Any],
    ) -> contextlib.AbstractContextManager:
        if override is None:
            return contextlib.nullcontext()

        img = end_points.get("img")
        if not torch.is_tensor(img) or img.dim() != 4:
            raise KeyError(
                "P0 geometry override requires end_points['img'] with shape "
                "[B,C,H,W]."
            )
        batch_size = int(img.shape[0])
        image_hw = (int(img.shape[-2]), int(img.shape[-1]))

        geometry_source = str(getattr(self, "geometry_depth_source", ""))
        if geometry_source == "pred":
            depth_net = getattr(self, "depth_net", None)
            if depth_net is None:
                raise AttributeError("P0 student has no depth_net module.")
            original_forward = depth_net.forward

            def _depth_forward(
                module_self: torch.nn.Module,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                del module_self
                result = original_forward(*args, **kwargs)
                return _replace_depth_net_output(
                    result,
                    override,
                    batch_size=batch_size,
                    image_hw=image_hw,
                )

            return _temporary_instance_method(depth_net, "forward", _depth_forward)

        if geometry_source == "gt":
            if not hasattr(self, "_prepare_gt_geometry_depth"):
                raise AttributeError(
                    "P0 teacher no longer exposes _prepare_gt_geometry_depth."
                )

            def _prepare_gt(
                model_self: torch.nn.Module,
                current_end_points: Mapping[str, Any],
                *,
                image_hw: Tuple[int, int],
                device: torch.device,
                dtype: torch.dtype,
            ) -> torch.Tensor:
                del model_self, current_end_points
                return _normalize_depth_override(
                    override,
                    batch_size=batch_size,
                    image_hw=(int(image_hw[0]), int(image_hw[1])),
                    device=device,
                    dtype=dtype,
                )

            return _temporary_instance_method(
                self,
                "_prepare_gt_geometry_depth",
                _prepare_gt,
            )

        raise P0RuntimeContractError(
            "P0 geometry override supports only geometry_depth_source='pred' "
            f"or 'gt', got {geometry_source!r}."
        )

    def _p0_configure_forced_query(
        self,
        end_points: MutableMapping[str, Any],
        forced_query: Optional[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if forced_query is None:
            return None
        img = end_points.get("img")
        if not torch.is_tensor(img):
            raise KeyError("P0 forced query requires end_points['img'].")
        contract = _canonicalize_query_contract(
            forced_query,
            device=img.device,
        )
        base = contract["kview_base_token_sel_idx"]
        token = contract["token_sel_idx"]
        view = contract["grasp_top_view_inds"]
        B, M = base.shape
        K = int(contract["kview_effective_k_int"])

        expected_m = int(getattr(self, "M_points", M))
        if M != expected_m:
            raise ValueError(
                "P0 forced-query base seed count differs from the current model: "
                f"forced M={M}, model M={expected_m}."
            )

        end_points["image_fps_seed_idx_override"] = base
        end_points.pop("oracle_view_inds_override", None)
        end_points.pop(P0B_EXACT_QUERY_VIEW_OVERRIDE_KEY, None)
        if K == 1:
            end_points["oracle_view_inds_override"] = view
        else:
            if bool(getattr(self, "is_training", False)):
                raise P0RuntimeContractError(
                    "Top-K P0 query replay requires the teacher to be "
                    "constructed with is_training=False."
                )
            install_p0b_exact_query_selector_override(self)
            end_points[P0B_EXACT_QUERY_VIEW_OVERRIDE_KEY] = (
                view.view(B, M, K).contiguous()
            )

        # Keep token only for assertion; the native image-FPS override recreates
        # it in the model.  No 3D center/view vector is injected here.
        del token
        return contract

    @staticmethod
    def _p0_assert_geometry(
        output: Mapping[str, Any],
        override: torch.Tensor,
    ) -> torch.Tensor:
        used = output.get("depth_map_used_for_geometry")
        if not torch.is_tensor(used):
            raise P0RuntimeContractError(
                "P0 model output is missing depth_map_used_for_geometry."
            )
        expected = override.detach()
        if expected.dim() == 3:
            expected = expected.unsqueeze(1)
        expected = expected.to(device=used.device, dtype=used.dtype)
        if tuple(used.shape) != tuple(expected.shape):
            raise P0RuntimeContractError(
                "P0 geometry override output shape mismatch: "
                f"used={tuple(used.shape)}, expected={tuple(expected.shape)}."
            )
        max_abs = (used.detach().float() - expected.float()).abs().max()
        if float(max_abs.item()) > 2.0e-5:
            raise P0RuntimeContractError(
                "P0 geometry intervention was not preserved exactly; "
                f"max_abs={float(max_abs.item()):.3e}."
            )
        return used

    def forward(
        self,
        end_points: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        if not isinstance(end_points, MutableMapping):
            raise TypeError(
                "P0 model forward expects a mutable endpoint mapping, got "
                f"{type(end_points)!r}."
            )
        # Use a shallow copy so the caller's top-level dictionary is not
        # rewritten.  This matches the normal endpoint semantics of the current
        # repository and leaves nested CPU label lists untouched.
        local: MutableMapping[str, Any] = dict(end_points)
        geometry_override = local.pop(P0_GEOMETRY_OVERRIDE_KEY, None)
        forced_query = local.pop(P0_FORCED_QUERY_KEY, None)
        query_contract = self._p0_configure_forced_query(local, forced_query)

        with self._p0_geometry_context(geometry_override, local):
            output = super().forward(local)
        if not isinstance(output, MutableMapping):
            raise TypeError(
                "Current EconomicGrasp P0 base model returned a non-mapping: "
                f"{type(output)!r}."
            )

        if geometry_override is not None:
            reference = self._p0_assert_geometry(output, geometry_override)
            output[P0_GEOMETRY_MARKER] = _scalar_like(reference, 1.0)
        if query_contract is not None:
            diagnostics = assert_p0_exact_query_output(query_contract, output)
            output.update(diagnostics)
            reference = output["token_sel_idx"]
            output[P0_QUERY_MARKER] = _scalar_like(reference, 1.0)
        return output


class economicgrasp_dpt_p0_student(_P0RuntimeMixin, _CurrentStudent):
    """Current Stage-1/2 RGB student plus P0-only interventions."""


class economicgrasp_dpt_p0_teacher(_P0RuntimeMixin, _CurrentTeacher):
    """Current Stage-0 clean-geometry teacher plus P0-only interventions."""


# Constructor aliases expected by the P0 repository adapter.
economicgrasp_dpt_student = economicgrasp_dpt_p0_student
economicgrasp_dpt_teacher = economicgrasp_dpt_p0_teacher


__all__ = [
    "DISTILL_CONTRACT_VERSION",
    "OutputDistillationConfig",
    "P0RuntimeContractError",
    "P0_RUNTIME_SCHEMA_VERSION",
    "P0_GEOMETRY_OVERRIDE_KEY",
    "P0_FORCED_QUERY_KEY",
    "P0_GEOMETRY_MARKER",
    "P0_QUERY_MARKER",
    "P0_GEOMETRY_MARKER_KEY",
    "P0_QUERY_MARKER_KEY",
    "runtime_contract",
    "extract_p0_query_contract",
    "build_p0_exact_query_input",
    "assert_p0_exact_query_output",
    "forward_with_p0_geometry_override",
    "enable_p0_exact_query_runtime",
    "compute_output_distillation_loss",
    "economicgrasp_dpt_p0_student",
    "economicgrasp_dpt_p0_teacher",
    "economicgrasp_dpt_student",
    "economicgrasp_dpt_teacher",
    "extract_distillation_targets",
    "load_checkpoint_state",
]
