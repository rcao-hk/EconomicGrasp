#!/usr/bin/env python3
"""Synthetic tests for scratch P2 variants; no GraspNet/DexNet required."""
from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.p2_gripper_cdf_field import (  # noqa: E402
    ACTION_POSE_DIM,
    P2_CACHE_SCHEMA_VERSION,
    P2_VARIANTS,
    RAY_FEATURE_DIM,
    GripperFieldSampler,
    P2FieldConfig,
    P2ScratchCdfMLP,
    active_evidence_blocks,
    build_action_pose_feature,
    projected_feature_dim,
)


def assert_close(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-6) -> None:
    error = float((a.float() - b.float()).abs().max().item())
    assert error <= tol, f"max error {error:.3e} exceeds {tol:.3e}"


def test_action_pose_periodicity() -> None:
    config = P2FieldConfig()
    center = torch.tensor([[0.05, -0.03, 0.62], [0.05, -0.03, 0.62]])
    view = torch.tensor([[0.2, 0.1, 0.97], [0.2, 0.1, 0.97]])
    angle = torch.tensor([0.37, 0.37 + torch.pi])
    depth = torch.tensor([0.02, 0.02])
    width = torch.tensor([0.06, 0.06])
    feature = build_action_pose_feature(center, view, angle, depth, width, config)
    assert feature.shape == (2, ACTION_POSE_DIM)
    assert_close(feature[0], feature[1], 2e-6)
    assert torch.isfinite(feature).all()
    assert config.sha256() == P2FieldConfig(**config.to_dict()).sha256()


def test_gripper_field_sampler() -> None:
    config = P2FieldConfig(residual_tau_m=0.02, surface_tau_m=0.01)
    sampler = GripperFieldSampler(config)
    assert sampler.num_samples == 32

    h = w = 448
    channels = 8
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    maps = [xx, yy, xx * yy, xx.square(), yy.square()]
    while len(maps) < channels:
        maps.append(torch.full_like(xx, float(len(maps)) / 10.0))
    image = torch.stack(maps[:channels], dim=0).unsqueeze(0)
    depth_map = torch.full((1, 1, h, w), 0.60)
    K = torch.tensor(
        [[[300.0, 0.0, 223.5], [0.0, 300.0, 223.5], [0.0, 0.0, 1.0]]]
    )
    center = torch.tensor([[0.0, 0.0, 0.60], [0.03, -0.02, 0.65]])
    rotation = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    width = torch.tensor([0.06, 0.08])
    grasp_depth = torch.tensor([0.02, 0.03])

    projected, ray, diagnostics = sampler(
        image,
        depth_map,
        K,
        center,
        rotation,
        width,
        grasp_depth,
        action_chunk=1,
    )
    assert projected.shape == (2, projected_feature_dim(channels))
    assert ray.shape == (2, RAY_FEATURE_DIM)
    assert torch.isfinite(projected).all() and torch.isfinite(ray).all()
    assert 0.0 < diagnostics["valid_ratio"] <= 1.0
    assert 0.0 < diagnostics["depth_valid_ratio"] <= 1.0
    assert float(ray.abs().sum().item()) > 0.0


def _all_variant_inputs(rows=7, depths=4, base_dim=16, image_dim=8):
    return {
        "base_feature": torch.randn(rows, base_dim),
        "action_pose_feature": torch.randn(rows, depths, ACTION_POSE_DIM),
        "projected_field_feature": torch.randn(
            rows, depths, projected_feature_dim(image_dim)
        ),
        "ray_depth_feature": torch.randn(rows, depths, RAY_FEATURE_DIM),
    }


def _kwargs_for_variant(inputs, variant):
    active = set(active_evidence_blocks(variant))
    kwargs = {}
    if "pose" in active:
        kwargs["action_pose_feature"] = inputs["action_pose_feature"]
    if "projected" in active:
        kwargs["projected_field_feature"] = inputs["projected_field_feature"]
    if "ray_depth" in active:
        kwargs["ray_depth_feature"] = inputs["ray_depth_feature"]
    return kwargs


def test_capacity_matched_three_layer_scratch_mlp() -> None:
    rows, depths, thresholds = 7, 4, 6
    base_dim, image_dim = 16, 8
    inputs = _all_variant_inputs(rows, depths, base_dim, image_dim)
    parameter_counts = []
    initial_state_dicts = []
    for variant in P2_VARIANTS:
        torch.manual_seed(123)
        model = P2ScratchCdfMLP(
            variant=variant,
            base_feature_dim=base_dim,
            image_feature_dim=image_dim,
            num_depths=depths,
            num_thresholds=thresholds,
            hidden_dim=32,
            increment_bias=-4.0,
        )
        linear_layers = [m for m in model.mlp.modules() if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 3
        assert model.contract()["uses_residual_on_stage1_or_p1"] is False
        assert model.contract()["scratch_initialization"] == "xavier_uniform"
        logits, raw = model(
            inputs["base_feature"], **_kwargs_for_variant(inputs, variant)
        )
        assert logits.shape == (rows, depths, thresholds)
        assert raw.shape == logits.shape
        assert torch.isfinite(logits).all()
        assert bool(((logits[..., 1:] - logits[..., :-1]) >= -1e-7).all())
        parameter_counts.append(sum(p.numel() for p in model.parameters()))
        initial_state_dicts.append(
            {key: value.detach().clone() for key, value in model.state_dict().items()}
        )

        target = torch.randint(0, 2, logits.shape).float()
        before = logits.detach().clone()
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        assert any(
            parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
            for parameter in model.parameters()
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        optimizer.step()
        after, _ = model(
            inputs["base_feature"], **_kwargs_for_variant(inputs, variant)
        )
        assert float((after - before).abs().max().item()) > 0.0

    # Fixed full input layout makes nominal capacity exactly equal, and a
    # shared seed gives bit-identical scratch initialization for every variant.
    assert len(set(parameter_counts)) == 1, parameter_counts
    reference_state = initial_state_dicts[0]
    for state in initial_state_dicts[1:]:
        assert set(state) == set(reference_state)
        for key in reference_state:
            assert_close(reference_state[key], state[key], 0.0)

    # Inactive evidence is hard-masked: passing arbitrary unused blocks cannot
    # alter P2-0 or P2-A outputs.
    torch.manual_seed(9)
    p20 = P2ScratchCdfMLP(
        variant="p2_0",
        base_feature_dim=base_dim,
        image_feature_dim=image_dim,
        num_depths=depths,
        num_thresholds=thresholds,
        hidden_dim=32,
    )
    clean, _ = p20(inputs["base_feature"])
    noisy, _ = p20(inputs["base_feature"], **{
        "action_pose_feature": inputs["action_pose_feature"] * 100.0,
        "projected_field_feature": inputs["projected_field_feature"] * 100.0,
        "ray_depth_feature": inputs["ray_depth_feature"] * 100.0,
    })
    assert_close(clean, noisy, 0.0)

    signature = inspect.signature(P2ScratchCdfMLP.forward)
    assert "p1_raw_cdf" not in signature.parameters
    assert "base_cdf_logits" not in signature.parameters


def _install_exact_action_stub() -> None:
    module = types.ModuleType("exact_action_cdf_common")
    module.CACHE_SCHEMA_VERSION = "cva_exact_action_cdf_head_cache_v1_1"
    module.CurrentCdfCheckpointContract = object
    module.sha256_file = lambda path: "stub-sha"
    module.validate_current_stage1_cdf_checkpoint = lambda *args, **kwargs: ({}, object())
    module.FRICTION_THRESHOLDS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)

    def atomic_save_json(payload, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    module.atomic_save_json = atomic_save_json
    sys.modules["exact_action_cdf_common"] = module


def _synthetic_cache_arrays(rows=3, depths=4, thresholds=6, feature_dim=16, image_dim=8):
    config = P2FieldConfig()
    friction = np.asarray(
        [
            [-1.0, 0.2, 0.6, 1.2],
            [0.4, 0.8, 1.0, -1.0],
            [0.2, 0.4, 0.6, 0.8],
        ],
        dtype=np.float32,
    )[:rows]
    collision = (friction < 0).astype(np.uint8)
    base_logits = np.random.randn(rows, depths, thresholds).astype(np.float32)
    return config, {
        "schema_version": np.asarray(P2_CACHE_SCHEMA_VERSION),
        "source_exact_action_cache_schema_version": np.asarray(
            "cva_exact_action_cdf_head_cache_v1_1"
        ),
        "source_base_checkpoint_sha256": np.asarray("base"),
        "field_config_json": np.asarray(config.canonical_json()),
        "field_config_sha256": np.asarray(config.sha256()),
        "cdf_increment_bias": np.asarray([-4.0], np.float32),
        "cdf_head_feature": np.random.randn(rows, feature_dim).astype(np.float32),
        "base_cdf_logits": base_logits,
        "action_pose_feature": np.random.randn(
            rows, depths, ACTION_POSE_DIM
        ).astype(np.float32),
        "projected_field_feature": np.random.randn(
            rows, depths, projected_feature_dim(image_dim)
        ).astype(np.float32),
        "ray_depth_feature": np.random.randn(
            rows, depths, RAY_FEATURE_DIM
        ).astype(np.float32),
        "friction": friction,
        "collision_or_empty": collision,
        "pure_collision": collision,
        "empty": np.zeros_like(collision),
        "assigned_obj": np.zeros((rows, depths), np.int16),
        "center_xyz": np.zeros((rows, 3), np.float32),
        "view_xyz": np.tile(
            np.asarray([[0.0, 0.0, 1.0]], np.float32), (rows, 1)
        ),
        "width_raw": np.zeros((rows, depths), np.float32),
        "center_id": np.arange(rows, dtype=np.int16),
        "angle_id": np.arange(rows, dtype=np.int8),
        "token_sel_idx": np.arange(rows, dtype=np.int32),
        "scene_id": np.asarray([0], np.int16),
        "anno_id": np.asarray([0], np.int16),
        "dataset_idx": np.asarray([0], np.int32),
        "feature_dim": np.asarray([feature_dim], np.int16),
        "image_feature_dim": np.asarray([image_dim], np.int16),
        "num_depths": np.asarray([depths], np.int16),
        "num_thresholds": np.asarray([thresholds], np.int16),
        "action_pose_dim": np.asarray([ACTION_POSE_DIM], np.int16),
        "projected_feature_dim": np.asarray(
            [projected_feature_dim(image_dim)], np.int32
        ),
        "ray_feature_dim": np.asarray([RAY_FEATURE_DIM], np.int16),
        "source_feature_max_abs": np.asarray([0.0], np.float32),
        "source_center_max_abs": np.asarray([0.0], np.float32),
        "source_view_max_abs": np.asarray([0.0], np.float32),
        "source_width_max_abs": np.asarray([0.0], np.float32),
        "base_endpoint_reconstruction_max_abs": np.asarray([0.0], np.float32),
        "source_base_logits_max_abs": np.asarray([0.0], np.float32),
        "field_valid_ratio": np.asarray([1.0], np.float32),
        "field_depth_valid_ratio": np.asarray([1.0], np.float32),
        "field_samples_per_action": np.asarray([32], np.int16),
    }


def test_cache_validator_and_row_dataset() -> None:
    _install_exact_action_stub()
    cache_module = importlib.import_module("p2_gripper_field_cache")
    rows, depths, feature_dim, image_dim = 3, 4, 16, 8
    config, arrays = _synthetic_cache_arrays(
        rows=rows, depths=depths, feature_dim=feature_dim, image_dim=image_dim
    )
    meta = cache_module.validate_p2_cache_arrays(
        arrays,
        expected_source_base_checkpoint_sha256="base",
        expected_field_config_sha256=config.sha256(),
    )
    assert meta.num_rows == rows and meta.num_depths == depths

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scene_0000" / "ann_0000.npz"
        path.parent.mkdir(parents=True)
        np.savez(path, **arrays)
        for variant in P2_VARIANTS:
            dataset = cache_module.P2GripperFieldCacheDataset(
                directory,
                split="all",
                variant=variant,
                expected_source_base_checkpoint_sha256="base",
                expected_field_config_sha256=config.sha256(),
            )
            item = dataset[0]
            assert item["friction"].shape == (rows, depths)
            batch = cache_module.collate_p2_gripper_field([item])
            assert batch["base_feature"].shape == (rows, feature_dim)
            assert batch["base_cdf_logits"].shape == (rows, depths, 6)
            assert batch["center_group"].shape == (rows,)
            active = set(active_evidence_blocks(variant))
            assert ("action_pose_feature" in batch) == ("pose" in active)
            assert ("projected_field_feature" in batch) == (
                "projected" in active
            )
            assert ("ray_depth_feature" in batch) == ("ray_depth" in active)


def test_predictor_checkpoint_roundtrip() -> None:
    _install_exact_action_stub()
    common = importlib.import_module("p2_gripper_field_common")
    config = P2FieldConfig()
    predictor = P2ScratchCdfMLP(
        variant="p2_c",
        base_feature_dim=16,
        image_feature_dim=8,
        num_depths=4,
        num_thresholds=6,
        hidden_dim=32,
        increment_bias=-4.0,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "predictor.tar")
        common.save_p2_predictor_checkpoint(
            path,
            predictor,
            variant="p2_c",
            epoch=3,
            train_metrics={"loss": 0.2},
            val_metrics={"loss": 0.3},
            source_base_checkpoint_sha256="base",
            field_config=config,
            cache_contract={"num_files": 2},
        )
        checkpoint, loaded, loaded_config = common.load_p2_predictor_checkpoint(
            path,
            expected_variant="p2_c",
            expected_source_base_checkpoint_sha256="base",
            expected_field_config=config,
        )
        assert int(checkpoint["epoch"]) == 3
        assert loaded.variant == "p2_c"
        assert loaded_config.sha256() == config.sha256()
        for key, value in predictor.state_dict().items():
            assert_close(value, loaded.state_dict()[key], 0.0)


def test_evaluation_protocol_helpers() -> None:
    _install_exact_action_stub()
    graspnet = types.ModuleType("graspnetAPI")
    graspnet.GraspNetEval = object
    sys.modules["graspnetAPI"] = graspnet
    module = importlib.import_module("eval_p2_gripper_cdf_field")

    reference = np.zeros((4, 2, 3, 6), dtype=np.float64)
    candidate = reference.copy()
    candidate[0] += 0.10
    candidate[1] += 0.20
    candidate[2] -= 0.05
    stats = module.paired_scene_statistics(
        reference, candidate, bootstrap_samples=1000, seed=0
    )
    assert stats["num_scenes"] == 4
    assert stats["improved_scenes"] == 2
    assert stats["degraded_scenes"] == 1
    assert stats["tied_scenes"] == 1
    assert (
        stats["bootstrap_ci95_low"]
        <= stats["mean_delta_ap"]
        <= stats["bootstrap_ci95_high"]
    )

    with tempfile.TemporaryDirectory() as directory:
        for mode in ("base", *P2_VARIANTS):
            for split in ("test_seen", "test_similar", "test_novel"):
                out = Path(directory) / mode / split
                out.mkdir(parents=True)
                payload = {
                    "status": "complete",
                    "mode": mode,
                    "test_mode": split,
                    "camera": "realsense",
                    "sample_interval": 0.1,
                    "collision_thresh": 0.01,
                    "collision_voxel_size": 0.01,
                    "top_views": 1,
                    "reference_base_checkpoint_sha256": "base",
                    "scratch_three_layer_mlp": mode != "base",
                    "uses_p1_checkpoint": False,
                    "uses_stage1_or_p1_residual": False,
                }
                (out / "_inference_complete.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
        manifests, lineage = module.validate_manifests(
            directory,
            ("base", *P2_VARIANTS),
            "realsense",
            10,
        )
        assert lineage == "base"
        assert len(manifests) == 15


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    tests = [
        test_action_pose_periodicity,
        test_gripper_field_sampler,
        test_capacity_matched_three_layer_scratch_mlp,
        test_cache_validator_and_row_dataset,
        test_predictor_checkpoint_roundtrip,
        test_evaluation_protocol_helpers,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"[PASS] {len(tests)}/{len(tests)} P2 synthetic tests")


if __name__ == "__main__":
    main()
