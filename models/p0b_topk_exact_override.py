"""P0-B-only exact K-view selector override.

The repository's normal Top-K inference selects the K highest-scoring views
independently for teacher and student. P0-B needs a stricter counterfactual:
the clean-depth teacher must evaluate exactly the student's selected K views,
in exactly the same center-major / rank-major order.

This module patches only the *teacher selector instance* used by P0-B. The
student and the repository-wide inference path are untouched.
"""

from __future__ import annotations

from types import MethodType
from typing import Any, Dict

import torch


P0B_EXACT_QUERY_VIEW_OVERRIDE_KEY = "p0b_exact_query_view_inds_override"


def install_p0b_exact_query_selector_override(model: torch.nn.Module) -> None:
    """Install an eval-only exact K-view override on one teacher model.

    The override tensor is read from
    ``end_points[P0B_EXACT_QUERY_VIEW_OVERRIDE_KEY]`` and must be [B,M,K].
    If the key is absent, the selector's original behavior is used unchanged.
    """
    grasp_module = getattr(model, "kview_grasp_module", None)
    if grasp_module is None:
        raise AttributeError(
            "P0-B Top-K override requires model.kview_grasp_module."
        )
    selector = getattr(grasp_module, "selector", None)
    if selector is None:
        raise AttributeError(
            "P0-B Top-K override requires model.kview_grasp_module.selector."
        )
    if bool(getattr(selector, "_p0b_exact_query_override_installed", False)):
        return

    original_forward = selector.forward

    def _forward_with_p0b_override(
        self,
        seed_features: torch.Tensor,
        seed_xyz: torch.Tensor,
        token_sel_idx: torch.Tensor,
        view_score: torch.Tensor,
        end_points: Dict[str, Any],
        is_training: bool,
        forced_view_inds=None,
    ):
        override = end_points.get(P0B_EXACT_QUERY_VIEW_OVERRIDE_KEY, None)
        if not torch.is_tensor(override):
            return original_forward(
                seed_features=seed_features,
                seed_xyz=seed_xyz,
                token_sel_idx=token_sel_idx,
                view_score=view_score,
                end_points=end_points,
                is_training=is_training,
                forced_view_inds=forced_view_inds,
            )

        if bool(is_training):
            raise RuntimeError(
                "P0-B exact K-view override is an evaluation-only mechanism."
            )

        view_score_bmv = self._normalize_view_score_shape(view_score)
        B, C, M = seed_features.shape
        if seed_xyz.shape[:2] != (B, M):
            raise ValueError(
                f"seed_xyz must be [B,M,3], got {tuple(seed_xyz.shape)}."
            )
        if token_sel_idx.shape != (B, M):
            raise ValueError(
                f"token_sel_idx must be [B,M], got {tuple(token_sel_idx.shape)}."
            )
        if view_score_bmv.shape[:2] != (B, M):
            raise ValueError(
                "view_score does not match the base seed shape: "
                f"score={tuple(view_score_bmv.shape)}, B/M={B}/{M}."
            )

        strategy, effective_k = self._strategy(is_training=False)
        if strategy not in {"top1", "topk"}:
            raise RuntimeError(
                "P0-B exact query override requires deterministic eval selection; "
                f"selector strategy={strategy!r}."
            )
        effective_k = min(int(effective_k), int(view_score_bmv.shape[-1]))
        override = override.detach().to(
            device=view_score_bmv.device,
            dtype=torch.long,
        )
        expected = (B, M, effective_k)
        if tuple(override.shape) != expected:
            raise ValueError(
                f"{P0B_EXACT_QUERY_VIEW_OVERRIDE_KEY} must be {expected}, "
                f"got {tuple(override.shape)}."
            )
        V = int(view_score_bmv.shape[-1])
        if bool(((override < 0) | (override >= V)).any()):
            raise ValueError(
                f"{P0B_EXACT_QUERY_VIEW_OVERRIDE_KEY} contains an out-of-range "
                f"view index for V={V}."
            )
        if effective_k > 1:
            sorted_override = torch.sort(override, dim=-1).values
            duplicate = (
                sorted_override[..., 1:] == sorted_override[..., :-1]
            ).any()
            if bool(duplicate):
                raise RuntimeError(
                    "P0-B exact Top-K override contains duplicate views within "
                    "one base center."
                )

        # Keep the teacher's original dense view field/probabilities. Only the
        # discrete selected query set is forced to the student's exact K views.
        view_prob = self._make_prob(view_score_bmv)
        view_inds = override.contiguous()
        view_rank = (
            torch.arange(
                effective_k,
                device=view_inds.device,
                dtype=torch.long,
            )
            .view(1, 1, effective_k)
            .expand(B, M, effective_k)
            .contiguous()
        )
        view_prob_sel = torch.gather(
            view_prob,
            dim=-1,
            index=view_inds,
        )

        Q = M * effective_k
        parent = (
            torch.arange(M, device=seed_features.device, dtype=torch.long)
            .view(1, M, 1)
            .expand(B, M, effective_k)
        )
        parent_q = parent.reshape(B, Q).contiguous()
        view_inds_q = view_inds.reshape(B, Q).contiguous()
        view_rank_q = view_rank.reshape(B, Q).contiguous()
        view_prob_q = view_prob_sel.reshape(B, Q).contiguous()

        seed_features_q = (
            seed_features.unsqueeze(-1)
            .expand(B, C, M, effective_k)
            .reshape(B, C, Q)
            .contiguous()
        )
        seed_xyz_q = (
            seed_xyz.unsqueeze(2)
            .expand(B, M, effective_k, 3)
            .reshape(B, Q, 3)
            .contiguous()
        )
        token_sel_idx_q = (
            token_sel_idx.unsqueeze(-1)
            .expand(B, M, effective_k)
            .reshape(B, Q)
            .contiguous()
        )
        view_xyz_q, view_rot_q = self._make_view_rot(view_inds_q)

        # Mirror the repository selector's endpoint contract exactly.
        end_points["kview_base_xyz_graspable"] = seed_xyz
        end_points["kview_base_token_sel_idx"] = token_sel_idx
        end_points["kview_base_seed_features"] = seed_features
        end_points["kview_base_M"] = int(M)
        end_points["kview_effective_k_int"] = int(effective_k)
        end_points["kview_mode"] = str(self.config.mode).upper()

        end_points["xyz_graspable"] = seed_xyz_q
        end_points["token_sel_xyz"] = seed_xyz_q
        end_points["token_sel_idx"] = token_sel_idx_q
        end_points["grasp_top_view_inds"] = view_inds_q
        end_points["grasp_top_view_xyz"] = view_xyz_q
        end_points["grasp_top_view_rot"] = view_rot_q

        end_points["kview_query_parent"] = parent_q
        end_points["kview_query_view_inds"] = view_inds_q
        end_points["kview_query_view_rank"] = view_rank_q
        end_points["kview_query_view_prob"] = view_prob_q
        end_points["kview_view_prob"] = view_prob.detach()

        self._add_debug(
            end_points,
            view_score_bmv.detach(),
            view_prob.detach(),
            view_inds.detach(),
            view_prob_sel.detach(),
            effective_k,
        )

        p = self.config.debug_prefix
        end_points[f"{p}Q forced view used"] = view_score_bmv.new_tensor(1.0).reshape(())
        end_points[f"{p}Q forced view diff"] = view_score_bmv.new_tensor(0.0).reshape(())
        end_points["D: P0B exact K-view override used"] = (
            view_score_bmv.new_tensor(1.0).reshape(())
        )
        end_points["D: P0B exact K-view override K"] = (
            view_score_bmv.new_tensor(float(effective_k)).reshape(())
        )
        return (
            seed_features_q,
            seed_xyz_q,
            token_sel_idx_q,
            view_rot_q,
            end_points,
        )

    selector.forward = MethodType(_forward_with_p0b_override, selector)
    selector._p0b_exact_query_override_installed = True
