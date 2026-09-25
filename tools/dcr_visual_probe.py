"""Reusable forward probe for DCR/E1 center-correction visualization."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch

from dcr_cva_common import make_outputs
from e1e2_common import select_centers
from models.economicgrasp_cva_centers import extract_depth_features, reference_candidates


@contextmanager
def preserve_eval(module):
    training = bool(module.training)
    try:
        module.eval()
        yield
    finally:
        module.train(training)


def subset_bundle(bundle: Dict[str, Any], query_ids: torch.Tensor) -> Dict[str, Any]:
    ids = query_ids.long()
    out = {}
    for k, v in bundle.items():
        if not torch.is_tensor(v):
            out[k] = v
        elif k in ("actions", "valid"):
            out[k] = v[:, ids]
        elif k in ("token_ids", "view_xyz", "angle_ids", "depth_ids", "query_ids", "native"):
            out[k] = v[ids]
        else:
            out[k] = v
    return out


@torch.no_grad()
def inspect_dcr_case(model, batch, case: str, case_seed: int,
                     query_limit: int = 0, query_chunk: int = 64,
                     depth_pack=None, local_queries: int = 8):
    """Return a rich, detached snapshot without changing the normal forward API."""
    pack = extract_depth_features(model.reference, batch) if depth_pack is None else depth_pack
    bundle, active_depth, stage1_ep = reference_candidates(
        model.reference, batch, pack, case, case_seed,
        model.corrector.offsets_mm.cpu().numpy(),
        query_limit=query_limit, return_end_points=True)

    h, w = batch["img"].shape[-2:]
    feature, proposal_logits, raw_feature, enhanced_feature, spatial_aux = (
        model.corrector.encode_image(
            batch, pack, active_depth, return_maps=True))

    local_ep: Dict[str, Any] = {}
    logits, latent = model.corrector.score_bundle(
        feature, proposal_logits, batch, active_depth, bundle,
        return_features=True, debug_sink=local_ep)
    native_score = bundle["actions"][model.zero, :, 0]
    residual = model.ranker(
        latent, logits, native_score,
        model.corrector.offsets_mm, model.zero)
    outputs, selected = make_outputs(
        logits, residual, bundle, model.zero, rank_strength=0.)

    utility = logits.sigmoid().mean(-1)
    stage1_order = torch.argsort(native_score, descending=True, stable=True)
    top_ids = stage1_order[:min(int(local_queries), len(stage1_order))]

    # Re-read only the highest Stage-1 queries so the grouping debug tensors
    # correspond to behavior a user will actually inspect, rather than the
    # arbitrary first chunk of the full query set.
    top_local_ep: Dict[str, Any] = {}
    if len(top_ids):
        sub = subset_bundle(bundle, top_ids)
        model.corrector.score_bundle(
            feature, proposal_logits, batch, active_depth, sub,
            return_features=False, debug_sink=top_local_ep)

    return {
        "pack": pack,
        "nominal_depth": pack[0],
        "active_depth": active_depth,
        "bundle": bundle,
        "stage1_end_points": stage1_ep,
        "feature_pre": raw_feature,
        "feature_post": enhanced_feature,
        "feature_fullres": feature,
        "proposal_logits": proposal_logits,
        "spatial_aux": spatial_aux,
        "cdf_logits": logits,
        "rank_residual": residual,
        "selected": selected,
        "selected_offsets_mm": model.corrector.offsets_mm[selected],
        "local_utility": utility,
        "outputs": outputs,
        "local_debug": top_local_ep if top_local_ep else local_ep,
        "top_query_indices": top_ids,
        "stage1_score": native_score,
    }


@torch.no_grad()
def inspect_e1_like_case(model, batch, bundle, active_depth, depth_pack,
                         query_chunk: int = 64):
    """Run an E1-compatible zero-ranker model on an already aligned DCR bundle."""
    feature, proposal_logits, raw, enhanced, spatial = model.corrector.encode_image(
        batch, depth_pack, active_depth, return_maps=True)
    logits = model.corrector(
        batch, bundle, case="nominal", case_seed=0,
        query_chunk=query_chunk, depth_pack=depth_pack)[0]
    utility = logits.sigmoid().mean(-1)
    selected = select_centers(utility, bundle["valid"], model.zero)
    return {
        "feature_pre": raw,
        "feature_post": enhanced,
        "proposal_logits": proposal_logits,
        "spatial_aux": spatial,
        "cdf_logits": logits,
        "selected": selected,
        "selected_offsets_mm": model.corrector.offsets_mm[selected],
        "local_utility": utility,
    }
