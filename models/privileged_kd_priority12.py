"""Priority-1/2 utilities for privileged clean-depth distillation.

This module is a direct overlay for the current EconomicGrasp GitHub revision
``52d3df2ba1f7d6c3ffd9cc74f6d976e6d15c4e08``.  It deliberately leaves the
canonical Stage-0/1/2 implementation untouched and adds two controlled
experiments:

1. Oracle teacher-better CDF hybrid diagnostics on exact student queries.
2. Exact-student-view teacher execution for View-only Stage-2 KD.

The exact-view training adapter is installed at runtime.  It performs a
seed-aligned teacher pass to preserve the teacher's unmodified dense view field,
then an exact-view pass for downstream query alignment.  This avoids changing
``models/economicgrasp_bip3d.py`` or distorting the View-KD target with the
bookkeeping logit boost used to pin an overridden view through downstream
argmax operations.
"""

from __future__ import annotations

import types
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import torch

from .privileged_kd_diagnostics import (
    _cdf_bins_to_target,
    _cdf_logits_to_qadt,
    _per_query_balanced_bce,
    _require_tensor,
    _safe_scalar_mean,
)


_EXACT_TEACHER_OVERRIDE_KEYS = (
    "image_fps_seed_idx_override",
    "oracle_view_inds_override",
)


def _as_bool(value: Any) -> bool:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return False
        return bool(value.detach().reshape(-1)[0].item())
    return bool(value)


def _detach_student_query_contract(
    student_end_points: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    """Extract the minimal exact-query contract without retaining activations."""
    required = (
        "kview_base_token_sel_idx",
        "token_sel_idx",
        "grasp_top_view_inds",
    )
    out: Dict[str, torch.Tensor] = {}
    for key in required:
        value = student_end_points.get(key)
        if not torch.is_tensor(value) or value.dim() != 2:
            raise TypeError(
                f"student_end_points[{key!r}] must be a rank-2 tensor; got "
                f"{type(value).__name__} with shape="
                f"{getattr(value, 'shape', None)}."
            )
        out[key] = value.detach().long()

    base_idx = out["kview_base_token_sel_idx"]
    query_idx = out["token_sel_idx"]
    view_idx = out["grasp_top_view_inds"]
    batch_size, num_base = base_idx.shape
    expected = (batch_size, num_base)
    if query_idx.shape != expected or view_idx.shape != expected:
        raise RuntimeError(
            "Exact student-view teacher execution currently supports one CVA "
            "view query per image-FPS seed (Q=M). Got "
            f"base={tuple(base_idx.shape)}, query={tuple(query_idx.shape)}, "
            f"view={tuple(view_idx.shape)}. Disable Top-K training or add a "
            "selector-level [B,M,K] view override."
        )
    if not torch.equal(query_idx, base_idx):
        raise RuntimeError(
            "Single-view exact-query training requires token_sel_idx to equal "
            "kview_base_token_sel_idx for every query."
        )
    return out


def build_exact_student_query_teacher_input(
    batch_input: Mapping[str, Any],
    student_query_contract: Mapping[str, torch.Tensor],
    *,
    force_process_grasp_labels: bool = False,
    compute_diagnostics: bool = False,
) -> Dict[str, Any]:
    """Build teacher input aligned to exact student seeds and view indices."""
    contract = _detach_student_query_contract(student_query_contract)
    teacher_input = dict(batch_input)
    for key in _EXACT_TEACHER_OVERRIDE_KEYS:
        teacher_input.pop(key, None)
    teacher_input["image_fps_seed_idx_override"] = contract[
        "kview_base_token_sel_idx"
    ]
    teacher_input["oracle_view_inds_override"] = contract[
        "grasp_top_view_inds"
    ]
    teacher_input["cva_force_process_grasp_labels"] = bool(
        force_process_grasp_labels
    )
    teacher_input["cva_compute_diagnostics"] = bool(compute_diagnostics)
    teacher_input["geometry_compute_diagnostics"] = False
    teacher_input["cva_export_angle_feature"] = False
    return teacher_input


def assert_exact_student_query_teacher_output(
    student_query_contract: Mapping[str, torch.Tensor],
    teacher_end_points: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    """Fail unless teacher/student base seeds, pixels, and views are identical."""
    contract = _detach_student_query_contract(student_query_contract)
    pairs = (
        ("kview_base_token_sel_idx", "base seed"),
        ("token_sel_idx", "expanded query pixel"),
        ("grasp_top_view_inds", "selected view"),
    )
    ratios: Dict[str, torch.Tensor] = {}
    for key, label in pairs:
        student = contract[key]
        teacher = teacher_end_points.get(key)
        if not torch.is_tensor(teacher):
            raise KeyError(
                f"Exact-query teacher output is missing tensor endpoint {key!r}."
            )
        teacher = teacher.to(device=student.device, dtype=student.dtype)
        if teacher.shape != student.shape:
            raise RuntimeError(
                f"Exact teacher/student {label} shapes differ: "
                f"{tuple(student.shape)} vs {tuple(teacher.shape)}."
            )
        equal = student == teacher
        if not bool(equal.all()):
            mismatch = float((~equal).float().mean().item())
            raise RuntimeError(
                f"Exact teacher/student {label} mismatch at "
                f"{100.0 * mismatch:.3f}% of entries."
            )
        ratios[key] = equal.float().mean().reshape(())
    return ratios


def install_exact_student_view_teacher_forward(
    trainer: Any,
    *,
    preserve_raw_view_field: bool = True,
    print_once: bool = True,
) -> None:
    """Attach exact-view teacher execution to an existing Stage-2 trainer.

    The current trainer already runs the student before the teacher.  A forward
    hook stores only the detached student query indices.  The teacher's forward
    method is then wrapped as follows:

    * pass 1: same student image-FPS seeds, no view override; retain the
      unmodified dense teacher ``view_score`` field;
    * pass 2: same seeds and exact student view indices; retain exact-view
      downstream CDF/width features and validate seed/pixel/view equality;
    * expose the raw pass-1 ``view_score`` in the returned exact-view endpoint
      dictionary so View-only KD does not imitate the artificial logit increase
      used internally to pin the overridden view.

    This adapter intentionally incurs a second frozen-teacher forward.  It is a
    controlled diagnostic/training experiment, not the final efficient method.
    """
    if getattr(trainer, "teacher", None) is None:
        raise RuntimeError(
            "Exact student-view teacher execution requires a Stage-2 trainer "
            "with a loaded frozen teacher."
        )
    if int(getattr(trainer, "distill_stage", -1)) != 2:
        raise RuntimeError(
            "Exact student-view teacher execution is defined only for Stage 2."
        )
    if getattr(trainer, "_priority12_exact_view_installed", False):
        return

    state: MutableMapping[str, Any] = {
        "student_query_contract": None,
        "verified_once": False,
    }

    def _capture_student_queries(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, Mapping):
            raise TypeError(
                "The Stage-2 student forward hook expected a mapping output, "
                f"got {type(output).__name__}."
            )
        state["student_query_contract"] = _detach_student_query_contract(output)

    hook_handle = trainer.net.register_forward_hook(_capture_student_queries)
    teacher = trainer.teacher
    original_forward = teacher.forward

    def _exact_forward(this: Any, batch_input: Mapping[str, Any]) -> Dict[str, Any]:
        del this  # ``original_forward`` is already bound to the teacher.
        contract = state.get("student_query_contract")
        if contract is None:
            raise RuntimeError(
                "The exact-view teacher was called before a student forward "
                "provided the current query contract."
            )

        force_labels = _as_bool(
            batch_input.get("cva_force_process_grasp_labels", False)
        )
        compute_diag = _as_bool(
            batch_input.get("cva_compute_diagnostics", False)
        )

        raw_view_score = None
        if preserve_raw_view_field:
            raw_input = dict(batch_input)
            raw_input.pop("oracle_view_inds_override", None)
            # Only the dense view field is required from this pass.
            raw_input["cva_force_process_grasp_labels"] = False
            raw_input["cva_compute_diagnostics"] = False
            raw_input["geometry_compute_diagnostics"] = False
            raw_input["cva_export_angle_feature"] = False
            raw_end_points = original_forward(raw_input)
            raw_view = raw_end_points.get("view_score")
            if not torch.is_tensor(raw_view):
                raise KeyError(
                    "Seed-aligned teacher pass did not export tensor 'view_score'."
                )
            raw_view_score = raw_view.detach().clone()
            del raw_end_points

        exact_input = build_exact_student_query_teacher_input(
            batch_input,
            contract,
            force_process_grasp_labels=force_labels,
            compute_diagnostics=compute_diag,
        )
        exact_end_points = original_forward(exact_input)
        ratios = assert_exact_student_query_teacher_output(
            contract, exact_end_points
        )

        if raw_view_score is not None:
            exact_view_score = exact_end_points.get("view_score")
            if not torch.is_tensor(exact_view_score):
                raise KeyError(
                    "Exact-view teacher pass did not export tensor 'view_score'."
                )
            if raw_view_score.shape != exact_view_score.shape:
                raise RuntimeError(
                    "Raw/exact teacher view-score shapes differ: "
                    f"{tuple(raw_view_score.shape)} vs "
                    f"{tuple(exact_view_score.shape)}."
                )
            exact_end_points["view_score_exact_override"] = exact_view_score
            exact_end_points["view_score"] = raw_view_score.to(exact_view_score)
            exact_end_points["D: Exact teacher raw view field preserved"] = (
                exact_view_score.new_tensor(1.0).reshape(())
            )

        ref = exact_end_points["grasp_top_view_inds"]
        exact_end_points["D: Exact teacher query enabled"] = (
            ref.new_tensor(1.0, dtype=torch.float32).reshape(())
        )
        exact_end_points["D: Exact teacher seed ratio"] = ratios[
            "kview_base_token_sel_idx"
        ]
        exact_end_points["D: Exact teacher pixel ratio"] = ratios[
            "token_sel_idx"
        ]
        exact_end_points["D: Exact teacher view ratio"] = ratios[
            "grasp_top_view_inds"
        ]

        if print_once and not bool(state["verified_once"]):
            rank = int(getattr(trainer, "rank", 0))
            if rank == 0:
                print(
                    "[PRIORITY12][EXACT-VIEW] verified teacher/student "
                    "seed=1.000, pixel=1.000, view=1.000; raw view field "
                    f"preserved={int(raw_view_score is not None)}",
                    flush=True,
                )
            state["verified_once"] = True
        return exact_end_points

    teacher._priority12_original_forward = original_forward
    teacher.forward = types.MethodType(_exact_forward, teacher)
    trainer._priority12_student_query_hook = hook_handle
    trainer._priority12_exact_view_installed = True


def _oracle_hybrid_variant_metrics(
    logits_bqadt: torch.Tensor,
    target_bqadt: torch.Tensor,
    valid_bqad: torch.Tensor,
    *,
    topk: Sequence[int],
    zero: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Evaluate one CDF-logit variant on the student's target/support."""
    balanced_loss, balanced_valid = _per_query_balanced_bce(
        logits_bqadt, target_bqadt, valid_bqad
    )
    prob = torch.sigmoid(logits_bqadt)
    pred_utility = prob.mean(dim=-1)  # [B,Q,A,D]
    target_utility = target_bqadt.mean(dim=-1)
    valid = valid_bqad.bool()
    batch_size, num_query, num_angle, num_depth = pred_utility.shape

    flat_pred = pred_utility.reshape(
        batch_size, num_query, num_angle * num_depth
    )
    flat_target = target_utility.reshape(
        batch_size, num_query, num_angle * num_depth
    )
    flat_valid = valid.reshape(
        batch_size, num_query, num_angle * num_depth
    )
    query_valid = flat_valid.any(dim=-1)

    masked_pred = flat_pred.masked_fill(~flat_valid, -1.0e6)
    selected_idx = masked_pred.argmax(dim=-1)
    selected_pred = flat_pred.gather(
        -1, selected_idx.unsqueeze(-1)
    ).squeeze(-1)
    selected_target = flat_target.gather(
        -1, selected_idx.unsqueeze(-1)
    ).squeeze(-1)
    oracle_target = flat_target.masked_fill(~flat_valid, -1.0).max(
        dim=-1
    ).values.clamp_min(0.0)
    regret = (oracle_target - selected_target).clamp_min(0.0)

    out = {
        "balanced BCE": _safe_scalar_mean(
            balanced_loss, balanced_valid, zero
        ),
        "selected predicted utility": _safe_scalar_mean(
            selected_pred, query_valid, zero
        ),
        "selected target utility": _safe_scalar_mean(
            selected_target, query_valid, zero
        ),
        "selection regret": _safe_scalar_mean(
            regret, query_valid, zero
        ),
        "exact-oracle ratio": _safe_scalar_mean(
            (regret <= 1.0e-6).float(), query_valid, zero
        ),
    }

    rank_score = selected_pred.masked_fill(~query_valid, -1.0e6)
    for requested_k in topk:
        requested_k = max(int(requested_k), 1)
        target_pieces = []
        regret_pieces = []
        for batch_i in range(batch_size):
            num_valid = int(query_valid[batch_i].sum().item())
            if num_valid <= 0:
                continue
            k_eff = min(requested_k, num_valid)
            idx = torch.topk(
                rank_score[batch_i], k=k_eff, largest=True
            ).indices
            target_pieces.append(selected_target[batch_i, idx].mean())
            regret_pieces.append(regret[batch_i, idx].mean())
        if target_pieces:
            out[f"top{requested_k} target utility"] = torch.stack(
                target_pieces
            ).mean().reshape(())
            out[f"top{requested_k} regret"] = torch.stack(
                regret_pieces
            ).mean().reshape(())
        else:
            out[f"top{requested_k} target utility"] = zero.reshape(())
            out[f"top{requested_k} regret"] = zero.reshape(())
    return out


@torch.no_grad()
def compute_oracle_teacher_better_hybrid_diagnostics(
    student_end_points: Mapping[str, Any],
    exact_view_teacher_end_points: Mapping[str, Any],
    *,
    teacher_better_margin: float = 0.0,
    topk: Sequence[int] = (10, 50),
    prefix: str = "D: PrivKD Oracle ",
) -> Dict[str, torch.Tensor]:
    """Evaluate exact-query teacher-CDF replacement variants.

    The oracle gate compares teacher and student threshold-balanced BCE on the
    *student* CDF target over common-valid angle-depth support.  The hybrid
    replaces teacher logits only for teacher-better queries and only inside
    common-valid bins. Student center, selected view, width, support, and all
    non-replaced CDF entries remain unchanged.
    """
    margin = float(teacher_better_margin)
    if margin < 0.0:
        raise ValueError("teacher_better_margin must be non-negative.")

    s_logits = _cdf_logits_to_qadt(
        _require_tensor(
            student_end_points, "grasp_cdf_pred_angle_depth"
        ).float()
    )
    t_logits = _cdf_logits_to_qadt(
        _require_tensor(
            exact_view_teacher_end_points,
            "grasp_cdf_pred_angle_depth",
        ).to(s_logits).float()
    )
    if s_logits.shape != t_logits.shape:
        raise ValueError(
            "Oracle hybrid requires exact-view teacher/student CDF shapes to "
            f"match, got {tuple(s_logits.shape)} vs {tuple(t_logits.shape)}."
        )

    s_bins = _require_tensor(
        student_end_points, "batch_grasp_cdf_bins_angle_depth"
    ).long().to(s_logits.device)
    s_valid = _require_tensor(
        student_end_points, "batch_grasp_cdf_valid_mask"
    ).bool().to(s_logits.device)
    t_valid = _require_tensor(
        exact_view_teacher_end_points, "batch_grasp_cdf_valid_mask"
    ).bool().to(s_logits.device)
    expected = s_logits.shape[:-1]
    if s_bins.shape != expected or s_valid.shape != expected:
        raise ValueError(
            "Student CDF target/valid shapes must be [B,Q,A,D], got "
            f"bins={tuple(s_bins.shape)}, valid={tuple(s_valid.shape)}, "
            f"expected={tuple(expected)}."
        )
    if t_valid.shape != expected:
        raise ValueError(
            "Teacher CDF valid shape differs from student query shape: "
            f"{tuple(t_valid.shape)} vs {tuple(expected)}."
        )

    target = _cdf_bins_to_target(
        s_bins, s_logits.shape[-1], s_logits.dtype
    )
    common_valid = s_valid & t_valid
    s_common_loss, s_common_query = _per_query_balanced_bce(
        s_logits, target, common_valid
    )
    t_common_loss, t_common_query = _per_query_balanced_bce(
        t_logits, target, common_valid
    )
    gate_valid = s_common_query & t_common_query
    teacher_better = gate_valid & (
        t_common_loss + margin < s_common_loss
    )

    common_mask = common_valid.unsqueeze(-1)
    hybrid_mask = (
        teacher_better.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        & common_mask
    )
    teacher_common = torch.where(common_mask, t_logits, s_logits)
    oracle_hybrid = torch.where(hybrid_mask, t_logits, s_logits)

    variants = {
        "student": s_logits,
        "teacher-full": t_logits,
        "teacher-common": teacher_common,
        "oracle-hybrid": oracle_hybrid,
    }
    zero = s_logits.new_zeros(())
    out: Dict[str, torch.Tensor] = {}
    for name, logits in variants.items():
        metrics = _oracle_hybrid_variant_metrics(
            logits,
            target,
            s_valid,
            topk=topk,
            zero=zero,
        )
        for metric_name, value in metrics.items():
            out[f"{prefix}{name} {metric_name}"] = value

    out[f"{prefix}common-valid ratio"] = (
        common_valid.float().mean().reshape(())
    )
    out[f"{prefix}gate-valid query ratio"] = (
        gate_valid.float().mean().reshape(())
    )
    out[f"{prefix}teacher-better query ratio"] = (
        teacher_better.float().mean().reshape(())
    )
    out[f"{prefix}teacher-better ratio among gate-valid"] = (
        _safe_scalar_mean(teacher_better.float(), gate_valid, zero)
    )
    out[f"{prefix}teacher common-BCE advantage"] = (
        _safe_scalar_mean(
            s_common_loss - t_common_loss, gate_valid, zero
        )
    )
    out[f"{prefix}hybrid replaced element ratio"] = (
        hybrid_mask.float().mean().reshape(())
    )
    out[f"{prefix}margin"] = zero.new_tensor(margin).reshape(())

    for metric_name in (
        "balanced BCE",
        "selected target utility",
        "selection regret",
        "top10 target utility",
        "top50 target utility",
    ):
        student_key = f"{prefix}student {metric_name}"
        hybrid_key = f"{prefix}oracle-hybrid {metric_name}"
        if student_key not in out or hybrid_key not in out:
            continue
        lower_is_better = "BCE" in metric_name or "regret" in metric_name
        if lower_is_better:
            improvement = out[student_key] - out[hybrid_key]
        else:
            improvement = out[hybrid_key] - out[student_key]
        out[f"{prefix}hybrid improvement {metric_name}"] = (
            improvement.reshape(())
        )
    return out
