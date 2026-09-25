#!/usr/bin/env python3
"""CPU checks of the actual E/Q/C boundaries with nonsaturated small tensors.

This does not replace the full-model, real-batch P0 audit. The two standalone
modules import normally; seed-selection/runtime methods are compiled from the
actual model's AST so this test needs no DINO weights or compiled CUDA FPS op.
The point-FPS check covers its small-candidate fallback, not the CUDA FPS kernel.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.grasp_spatial_enhancer import GraspSpatialEnhancer
from models.kview_query_transformer import (
    KViewQueryTransformerConfig,
    ViewConditionedAttentionGrouping,
)


def model_methods():
    path = ROOT / "models" / "economicgrasp_bip3d.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == "economicgrasp_dpt")
    wanted = {
        "set_depth_grad_routes", "get_depth_grad_routes",
        "_select_graspable_seed_queries", "_validate_image_fps_override",
        "_backproject_uvz",
    }
    model.body = [n for n in model.body if isinstance(n, ast.FunctionDef)
                  and n.name in wanted]
    assert {n.name for n in model.body} == wanted
    module = ast.fix_missing_locations(ast.Module(body=[model], type_ignores=[]))
    namespace = dict(globals())
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["economicgrasp_dpt"]


def connected(loss, depth, expected):
    # A trainable readout keeps loss differentiable when the boundary is closed.
    readout = torch.ones((), requires_grad=True)
    grad, = torch.autograd.grad(loss * readout, depth, allow_unused=True)
    if expected:
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().max().item() > 0.0
    else:
        assert grad is None, "closed boundary must be unused, not connected_zero"
    return None if grad is None else float(grad.norm())


def equal(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def main():
    torch.manual_seed(619)
    model = model_methods()()
    enhancer = GraspSpatialEnhancer(embed_dims=8, feature_3d_dim=4, vis_dir=None)
    config = KViewQueryTransformerConfig(
        grouping_model_dim=8, grouping_num_heads=2, grouping_dropout=0.0,
        patch_size=2, radius_px_min=0.1, radius_px_max=2.0,
        metric_radius=0.01,
    )
    group = ViewConditionedAttentionGrouping(8, 8, 8, config)
    model.spatial_enhancer = enhancer
    model.kview_config = config
    model.kview_grasp_module = SimpleNamespace(config=config, group=group)
    model.is_training = True
    model.use_gt_xyz_for_train = False
    model.M_points = 2
    model.min_depth, model.max_depth = 0.2, 1.0

    route_cases = {
        "none": (False, False, False), "all": (True, True, True),
        "gse": (True, False, False), "seed_xyz": (False, True, False),
        "support": (False, False, True),
        "support,gse": (True, False, True),
        "seed_xyz,support": (False, True, True),
        "gse,seed_xyz": (True, True, False),
    }
    for route, flags in route_cases.items():
        model.set_depth_grad_routes(route)
        assert tuple(model.get_depth_grad_routes().values()) == flags
        assert config.detach_aux_maps, "auxiliary map stop-gradient was changed"
    for bad in ("", "none,gse", "all,support", "depth", "gse,", []):
        try:
            model.set_depth_grad_routes(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid route {bad!r}")
    assert model.set_depth_grad_routes(["support", "gse"]) == "gse,support"

    depth0 = torch.linspace(0.45, 0.60, 16).reshape(1, 1, 4, 4)
    feat = torch.randn(1, 8, 4, 4)
    K = torch.tensor([[[20.0, 0.0, 1.5], [0.0, 20.0, 1.5], [0.0, 0.0, 1.0]]])
    rows = []

    reference = None
    for route in ("none", "gse"):
        model.set_depth_grad_routes(route)
        depth = depth0.clone().requires_grad_()
        out, aux = enhancer(feat, depth_map=depth, K=K, image_hw=(4, 4),
                            capture_depth_grad=True)
        if reference is None:
            reference = out.detach()
        equal(out, reference)
        assert aux["depth_grad_gse_input"].requires_grad == (route == "gse")
        norm = connected(out.square().mean(), depth, route == "gse")
        rows.append({"path": "gse", "route": route, "grad_norm": norm})
    _, plain_aux = enhancer(feat, depth_map=depth0, K=K, image_hw=(4, 4))
    assert "depth_grad_gse_input" not in plain_aux

    for mode in ("image_fps", "point_fps"):
        model.seed_selection_mode = mode
        reference = None
        for route in ("none", "seed_xyz"):
            model.set_depth_grad_routes(route)
            depth = depth0.clone().requires_grad_()
            # Point-FPS fallback has one candidate, avoiding the CUDA-only op.
            mask = torch.zeros(1, 16, dtype=torch.bool)
            mask[:, 5] = True
            ep = {"depth_grad_capture_routes": True}
            if mode == "image_fps":
                ep["image_fps_seed_idx_override"] = torch.tensor([[5, 10]])
            result = model._select_graspable_seed_queries(
                feat, depth, K, mask, mask, torch.ones(1, 16), ep)
            seed_xyz = result[1]
            if reference is None:
                reference = tuple(x.detach().clone() for x in result[:3])
            for actual, expected in zip(result[:3], reference):
                equal(actual, expected)
            assert ep["depth_grad_seed_xyz_input"].requires_grad == (route == "seed_xyz")
            norm = connected(seed_xyz.square().mean(), depth, route == "seed_xyz")
            rows.append({"path": mode, "route": route, "grad_norm": norm})

    # Direct C support-map derivatives, with fixed Q/grid.
    grid = torch.tensor([[[[-0.5, -0.5], [0.5, 0.5]]]])
    seed_xyz = torch.tensor([[[0.0, 0.0, 0.52]]])
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    reference = None
    for route in ("none", "support"):
        model.set_depth_grad_routes(route)
        depth = depth0.clone().requires_grad_()
        aliases = []
        delta, *_ = group._sample_aux_maps(grid, seed_xyz, depth, None, None,
                                          valid, depth_grad_inputs=aliases)
        if reference is None:
            reference = delta.detach()
        equal(delta, reference)
        assert len(aliases) == 1 and aliases[0].requires_grad == (route == "support")
        norm = connected(delta.sum(), depth, route == "support")
        rows.append({"path": "support", "route": route, "grad_norm": norm})

    # C being closed must NOT block Q's grid/radius or residual-center gradient.
    model.set_depth_grad_routes("seed_xyz")
    center = torch.tensor([[[0.0, 0.0, 0.52]]], requires_grad=True)
    rot = torch.eye(3).reshape(1, 1, 3, 3)
    native_grid, native_valid, *_ = group._make_view_conditioned_grid(
        center, torch.tensor([[5]]), rot, depth0, K, 4, 4)
    delta, *_ = group._sample_aux_maps(native_grid, center, depth0, None, None,
                                     native_valid)
    center_grad, = torch.autograd.grad(delta.sum(), center)
    assert center_grad[..., 2].abs().max().item() > 0.0
    rows.append({"path": "Q_with_C_closed", "grad_norm": float(center_grad.norm())})
    assert all(p.grad is None for p in enhancer.parameters())
    assert all(p.grad is None for p in group.parameters())
    print(json.dumps({"status": "passed", "scope": "synthetic boundary checks",
                      "results": rows}, indent=2))


if __name__ == "__main__":
    main()
