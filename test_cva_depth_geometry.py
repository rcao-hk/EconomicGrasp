"""CPU regression tests; no dataset, checkpoint or compiled CUDA extensions.

The CVA integration tests use the real selector, grouping and CDF decoder with
synthetic ViewNet outputs and cached-label fixtures. They do not stand in for
the full DINO/dataset/CUDA smoke run documented in DEPTH_GEOMETRY_EXPERIMENTS.md.
"""

import importlib.util
from datetime import timedelta
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cva_depth_geometry import (PairConfig, contrast_depth, depth_metrics, matched_depth_pairs,
                                metric_depth_loss, pair_metrics, relative_depth_loss, visible_anchor_pool)
from cva_depth_experiment import (CPU_LABEL_KEYS, FixedSupportReplay, assert_cpu_label_residency,
                                  checkpoint_metadata, fixed_anchor_seeds, configure_repository,
                                  grasp_objective, move_batch, select_frame_indices)
import train_cva_depth_geometry as training
from diagnose_cva_depth_contrast import summarize


ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)


def load_cpu_module(name, relative_path):
    # models/__init__.py eagerly imports MinkowskiEngine. Load the actual
    # standalone source file without initializing that unrelated package.
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_batch():
    h, w = 33, 41
    v, u = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    gt = (0.50 + u * 0.001 + v * 0.0007)[None, None]
    k = torch.tensor([[100., 0., 20.], [0., 100., 16.], [0., 0., 1.]])[None]
    pixels = torch.tensor([10 * w + 10, 12 * w + 16, 19 * w + 21, 23 * w + 29])
    z = gt.flatten()[pixels]
    xyz = torch.stack(((pixels % w - 20) * z / 100, (pixels // w - 16) * z / 100, z), -1)
    pose = torch.tensor([[0., -1., 0., .013], [1., 0., 0., -.017], [0., 0., 1., .03]])
    points = (xyz - pose[:, 3]) @ pose[:, :3]
    return {"gt_depth_m": gt, "K": k, "img": torch.cat((gt, gt, gt), 1),
            "objectness_label_tok": torch.ones(1, h * w, dtype=torch.long),
            "grasp_points_list": [[points[:2], points[2:]]],
            "object_poses_list": [[pose.clone(), pose.clone()]],
            "dataset_idx": torch.tensor([7]), "scene_idx": torch.tensor([0]), "anno_idx": torch.tensor([7])}


def training_args(**kwargs):
    args = training.parse_args(["--dataset_root", "fixture", "--checkpoint_path", "fixture.tar",
                                "--output_dir", "unused"])
    args.anchors_per_image, args.pairs_per_anchor = 16, 4
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def cpu_ddp_label_worker(rank, rendezvous, result_dir):
    """Real two-rank DDP regression for nested labels; no CUDA model imports."""
    dist = torch.distributed
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        inputs = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10
        # Four samples per rank, with two variable-length object tensors each.
        labels = {key: [[torch.full((sample_i + 2, 3), float(rank)),
                         torch.full((sample_i + 3, 3), float(rank))]
                        for sample_i in range(4)] for key in CPU_LABEL_KEYS}
        batch = move_batch({"img": inputs + rank, **labels}, torch.device("cpu"))

        class LabelProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([.2, -.3]))
                self.unused = torch.nn.Parameter(torch.tensor(1.))
                self.checked_labels = 0

            def forward(self, current):
                assert_cpu_label_residency(current)
                for key in CPU_LABEL_KEYS:
                    for sample_i, objects in enumerate(current[key]):
                        for object_i, value in enumerate(objects):
                            if value is not labels[key][sample_i][object_i]:
                                raise AssertionError("DDP must preserve the original CPU cache tensors.")
                            self.checked_labels += 1
                targets = torch.stack([objects[0][0, 0] for objects in current["view_graspness_list"]])
                return (current["img"] @ self.weight - targets).square().mean()

        module = LabelProbe()
        network = training.build_training_network(module, world_size=2)
        optimizer = torch.optim.SGD(network.parameters(), lr=.01)
        all_inputs = torch.cat((inputs, inputs + 1))
        all_targets = torch.cat((torch.zeros(4), torch.ones(4)))
        # Catch any future recursive input-copy path, in addition to verifying
        # actual forward/backward synchronization over two Gloo processes.
        with patch("torch.nn.parallel.distributed._to_kwargs",
                   side_effect=AssertionError("DDP must not recursively transfer the cache")):
            for _ in range(2):
                reference = module.weight.detach().clone().requires_grad_(True)
                reference_loss = (all_inputs @ reference - all_targets).square().mean()
                expected_grad, = torch.autograd.grad(reference_loss, reference)
                optimizer.zero_grad(set_to_none=True)
                loss = network(batch)
                loss.backward()
                torch.testing.assert_close(module.weight.grad, expected_grad)
                if module.unused.grad is not None:
                    raise AssertionError("Unused parameter unexpectedly received a gradient.")
                optimizer.step()
                torch.testing.assert_close(module.weight, reference.detach() - .01 * expected_grad)
        (Path(result_dir) / f"rank_{rank}.json").write_text(json.dumps(
            {"rank": rank, "checked_labels": module.checked_labels, "weight": module.weight.detach().tolist()}
        ), encoding="utf-8")
    finally:
        dist.destroy_process_group()


class DistributedInputTests(unittest.TestCase):
    def test_explicit_transfer_preserves_cpu_cache(self):
        batch = synthetic_batch()
        batch["view_graspness_list"] = [[torch.ones(2, 8), torch.ones(3, 8)]]
        # meta exercises a non-CPU dense-input transfer without needing a GPU.
        moved = move_batch(batch, torch.device("meta"))
        self.assertEqual(moved["img"].device.type, "meta")
        for key in CPU_LABEL_KEYS.intersection(batch):
            self.assertIs(moved[key], batch[key])
        assert_cpu_label_residency(moved)

    def test_model_boundary_rejects_a_moved_cache_before_forward(self):
        batch = synthetic_batch()
        batch["view_graspness_list"] = [[torch.empty(2, 8, device="meta")]]
        model = torch.nn.Module()
        module = training.DepthTrainingModule(model, training_args(), None)
        with patch.object(model, "forward") as forward:
            with self.assertRaisesRegex(RuntimeError, r"view_graspness_list\[0\]\[0\].*device_ids=None"):
                module(batch, pair_seed=0)
            forward.assert_not_called()

    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(),
                         "Two-process CPU regression requires Gloo.")
    def test_two_rank_ddp_preserves_cache_and_synchronizes_gradients(self):
        with tempfile.TemporaryDirectory(prefix="cva_depth_ddp_") as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            torch.multiprocessing.spawn(cpu_ddp_label_worker, args=(rendezvous, directory), nprocs=2, join=True)
            results = [json.loads((Path(directory) / f"rank_{rank}.json").read_text(encoding="utf-8"))
                       for rank in range(2)]
            self.assertEqual([r["rank"] for r in results], [0, 1])
            self.assertTrue(all(r["checked_labels"] == 2 * len(CPU_LABEL_KEYS) * 4 * 2 for r in results))
            self.assertEqual(results[0]["weight"], results[1]["weight"])


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.batch = synthetic_batch()
        self.cfg = PairConfig(anchors_per_image=32, pairs_per_anchor=4)
        self.gt = self.batch["gt_depth_m"]

    def test_camera_transform_visibility_and_duplicate_pixels(self):
        pool = visible_anchor_pool(self.batch, 0, self.cfg)
        self.assertEqual(pool["pixel"].tolist(), [420, 508, 800, 972])
        original = pool["xyz"].clone()
        pose = self.batch["object_poses_list"][0][0]
        # Same image ray, a less accurate duplicate and an occluded anchor.
        bad = torch.stack((original[0] * ((original[0, 2] + .003) / original[0, 2]),
                           original[1] * ((original[1, 2] + .02) / original[1, 2]),
                           torch.tensor([10., 0., .5]), torch.tensor([float("nan"), 0., .5])))
        bad = (bad - pose[:, 3]) @ pose[:, :3]
        self.batch["grasp_points_list"][0][0] = torch.cat((self.batch["grasp_points_list"][0][0], bad))
        checked = visible_anchor_pool(self.batch, 0, self.cfg)
        torch.testing.assert_close(checked["xyz"], original)
        self.batch["objectness_label_tok"][0, 420] = 0
        self.assertEqual(len(visible_anchor_pool(self.batch, 0, self.cfg)["pixel"]), 3)

    def test_pair_sampling_is_gt_only_and_matched(self):
        pairs = matched_depth_pairs(self.batch, self.cfg, 13)
        changed = dict(self.batch, depth_map_pred=torch.rand_like(self.gt), graspness_score=torch.randn(1, 1, self.gt.numel()))
        again = matched_depth_pairs(changed, self.cfg, 13)
        self.assertGreater(int(pairs["valid"].sum()), 100)
        for key in ("anchor_i", "anchor_j", "foreground_i", "foreground_j", "valid"):
            self.assertTrue(torch.equal(pairs[key], again[key]))
        mask, width = pairs["valid"], self.gt.shape[-1]
        for operation in (lambda x: x % width, lambda x: x // width):
            a = operation(pairs["anchor_i"]) - operation(pairs["anchor_j"])
            f = operation(pairs["foreground_i"]) - operation(pairs["foreground_j"])
            self.assertTrue(torch.equal(a[mask], f[mask]))
        z = self.gt.flatten(1)
        dz = z.gather(1, pairs["anchor_i"]) - z.gather(1, pairs["foreground_i"])
        self.assertLessEqual(float(dz[mask].abs().max()), self.cfg.control_depth_tolerance_m)
        pm = pair_metrics(self.gt, self.gt, pairs, self.cfg, self.batch["K"])
        self.assertEqual(pm["anchor_relative_mae_m"], 0)
        self.assertGreater(pm["anchor_mean_lateral_radius_m"], 0)

    def test_relative_loss_is_shift_invariant_and_restores_shape(self):
        pairs = matched_depth_pairs(self.batch, self.cfg, 9)
        for variant in ("foreground", "anchor"):
            self.assertEqual(float(relative_depth_loss(self.gt, self.gt, pairs, variant, self.cfg)), 0)
            shifted = relative_depth_loss(self.gt + .08, self.gt, pairs, variant, self.cfg)
            self.assertLess(float(shifted), 1e-10)
            flat = torch.full_like(self.gt, float(self.gt.mean()), requires_grad=True)
            loss = relative_depth_loss(flat, self.gt, pairs, variant, self.cfg)
            self.assertGreater(float(loss.detach()), 1e-5)
            gradient, = torch.autograd.grad(loss, flat)
            self.assertAlmostEqual(float(gradient.sum()), 0, places=6)
            self.assertGreater(float((gradient * (flat.detach() - self.gt)).sum()), 0)
            after = relative_depth_loss(flat.detach() - .01 * gradient, self.gt, pairs, variant, self.cfg)
            self.assertLess(float(after), float(loss.detach()))
        self.assertGreater(float(metric_depth_loss(self.gt + .08, self.gt, self.cfg)), .07)

    def test_arbitrary_variance_is_not_rewarded(self):
        pairs = matched_depth_pairs(self.batch, self.cfg, 4)
        wrong = 2 * self.gt.mean() - self.gt
        torch.testing.assert_close(wrong.std(), self.gt.std())
        self.assertGreater(float(relative_depth_loss(wrong, self.gt, pairs, "anchor", self.cfg)), 0)

    def test_empty_pairs_and_nan_gt_have_connected_zero_loss(self):
        self.batch["gt_depth_m"] = torch.full_like(self.gt, float("nan"))
        pairs = matched_depth_pairs(self.batch, self.cfg, 3)
        pred = self.gt.clone().requires_grad_(True)
        loss = relative_depth_loss(pred, self.batch["gt_depth_m"], pairs, "anchor", self.cfg)
        loss = loss + metric_depth_loss(pred, self.batch["gt_depth_m"], self.cfg)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))
        self.assertIsNone(depth_metrics(pred, self.batch["gt_depth_m"], self.cfg)["std_ratio"])

    def test_metric_mask_and_original_normalization(self):
        gt = torch.tensor([[[[.5, 0., float("nan"), 2.]]]])
        pred = torch.tensor([[[[.6, .7, .9, .4]]]], requires_grad=True)
        loss = metric_depth_loss(pred, gt, self.cfg)
        self.assertAlmostEqual(float(loss.detach()), .025, places=6)
        self.assertAlmostEqual(float(metric_depth_loss(pred, gt, self.cfg, "valid_pixels").detach()), .1, places=6)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([[[[.25, 0., 0., 0.]]]]))

    def test_contrast_mean_identity_no_clipping(self):
        depth = torch.tensor([[[[.2, .4, .8, 1.]]]])
        torch.testing.assert_close(contrast_depth(depth, 1), depth)
        torch.testing.assert_close(contrast_depth(depth, 0), depth.mean().expand_as(depth))
        exaggerated = contrast_depth(depth, 2)
        self.assertGreater(float(exaggerated.max()), 1)
        torch.testing.assert_close(exaggerated.mean(), depth.mean())
        with self.assertRaises(ValueError):
            contrast_depth(depth, -1)

    def test_imagewise_std_detects_planes_with_different_means(self):
        gt = self.gt.repeat(2, 1, 1, 1)
        pred = torch.stack((torch.full_like(self.gt[0], .4), torch.full_like(self.gt[0], .7)))
        self.assertGreater(float(pred.std()), .1)
        metrics = depth_metrics(pred, gt, self.cfg)
        self.assertLess(metrics["pred_std_m"], 1e-6)
        self.assertLess(metrics["std_ratio"], 1e-4)


class FixedQueryTests(unittest.TestCase):
    def test_fixed_seed_values_are_owned_and_features_still_differentiable(self):
        model = SimpleNamespace(_select_graspable_seed_queries=lambda **kwargs: "original")
        xyz = torch.tensor([[[.1, .2, .5], [.2, .3, .6]]])
        pixel = torch.tensor([[1, 5]])
        feat = torch.arange(24.).reshape(1, 2, 3, 4).requires_grad_(True)
        original_xyz = xyz.clone()
        with fixed_anchor_seeds(model, xyz, pixel):
            xyz.zero_()
            pixel.zero_()
            output = model._select_graspable_seed_queries(feat_grid=feat, depth_map=torch.zeros(1, 1, 3, 4),
                      camera_K=None, graspable_mask=None, valid_tok=None, grasp_score=None, end_points={})
            torch.testing.assert_close(output[1], original_xyz)
            self.assertEqual(output[2].tolist(), [[1, 5]])
            self.assertEqual(output[-1], 2)
            output[0].sum().backward()
            self.assertEqual(int((feat.grad != 0).sum()), 4)
        self.assertEqual(model._select_graspable_seed_queries(), "original")

    def test_actual_cva_replay_and_depth_input_gradient(self):
        cva_source = load_cpu_module("_depth_test_cva", "models/kview_query_transformer.py")
        override_source = load_cpu_module("_depth_test_override", "models/p0b_topk_exact_override.py")
        from utils.loss_utils import batch_viewpoint_params_to_matrix, generate_grasp_views

        class SyntheticView(torch.nn.Module):
            def forward(self, seed_features, end_points, **kwargs):
                b, _, q = seed_features.shape
                scores = torch.linspace(-1, 1, 8).view(1, 1, 8).expand(b, q, 8).clone()
                end_points["view_score"] = scores
                end_points["grasp_top_view_inds"] = scores.argmax(-1)
                return end_points, torch.zeros_like(seed_features)

        torch.manual_seed(2)
        config = cva_source.KViewQueryTransformerConfig(mode="A1", patch_size=3, metric_radius=.01,
                  radius_px_min=1, radius_px_max=5, grouping_model_dim=32, head_model_dim=32,
                  grouping_max_queries_per_chunk=7, grouping_dropout=0, head_dropout_p=0, head_attn_dropout=0)
        cva = cva_source.CenterViewAngleQueryTransformerLocalGraspModule(SyntheticView(), num_view=8, num_angle=12,
                  num_depth=4, seed_feature_dim=8, feat_dim=8, view_dirs=generate_grasp_views(8),
                  batch_viewpoint_params_to_matrix_fn=batch_viewpoint_params_to_matrix, config=config, use_cdf=True)
        model = SimpleNamespace(kview_grasp_module=cva)
        cva.eval()
        override_source.install_p0b_exact_query_selector_override(model)
        b = synthetic_batch()
        pool = visible_anchor_pool(b, 0, PairConfig())
        xyz, pixels, views = pool["xyz"][:2][None], pool["pixel"][:2][None], torch.tensor([[2, 4]])
        features = torch.randn(1, 8, 33, 41)
        seeds = features.flatten(2).gather(2, pixels[:, None].expand(-1, 8, -1))
        label_calls = []

        def label_fixture(ep, **kwargs):
            label_calls.append(ep["grasp_top_view_inds"].clone())
            q = ep["xyz_graspable"].shape[1]
            ep.update(batch_grasp_point=ep["xyz_graspable"].clone(),
                      batch_grasp_view_graspness=torch.linspace(0, 1, 8).view(1, 1, 8).expand(1, q, 8).clone(),
                      batch_grasp_cdf_bins_angle_depth=torch.ones(1, q, 12, 4, dtype=torch.long),
                      batch_grasp_cdf_valid_mask=torch.ones(1, q, 12, 4, dtype=torch.bool),
                      batch_grasp_width_angle_depth=torch.full((1, q, 12, 4), .04),
                      batch_grasp_width_valid_mask_angle_depth=torch.ones(1, q, 12, 4, dtype=torch.bool),
                      batch_grasp_cdf_thresholds=torch.arange(1, 7) * .2)
            return torch.eye(3).expand(1, q, 3, 3), ep

        def forward(depth, seed_xyz=xyz):
            return cva(seed_features=seeds, seed_xyz=seed_xyz, token_sel_idx=pixels, feat_map=features,
                       depth_map=depth, camera_K=b["K"], is_training=False,
                       process_grasp_labels_fn=label_fixture,
                       end_points={override_source.P0B_EXACT_QUERY_VIEW_OVERRIDE_KEY: views[..., None],
                                   "cva_compute_diagnostics": False})

        original_grid = cva.group._make_view_conditioned_grid
        with FixedSupportReplay(model, views) as replay:
            replay.begin(recording=True)
            with torch.no_grad():
                reference = forward(b["gt_depth_m"])
            replay.finish()
            self.assertEqual(len(label_calls), 2)
            self.assertTrue(all(torch.equal(v, views) for v in label_calls))
            self.assertGreater(len(replay.grids), 1)  # Exercise chunked grid replay.
            fingerprint = replay.fingerprint()
            for beta in (0, .5, 1.25):
                replay.begin()
                with torch.no_grad():
                    ep = forward(contrast_depth(b["gt_depth_m"], beta))
                replay.finish()
                self.assertEqual(len(label_calls), 2)
                self.assertEqual(fingerprint, replay.fingerprint())
                for key in ("xyz_graspable", "token_sel_idx", "grasp_top_view_inds", "batch_grasp_cdf_bins_angle_depth",
                            "batch_grasp_cdf_valid_mask", "batch_grasp_view_graspness"):
                    self.assertTrue(torch.equal(ep[key], reference[key]), key)
            self.assertEqual(tuple(ep["grasp_cdf_pred_angle_depth"].shape), (1, 6, 2, 12, 4))
            replay.begin()
            depth = b["gt_depth_m"].clone().requires_grad_(True)
            ep = forward(depth)
            gradient, = torch.autograd.grad(ep["grasp_cdf_pred_angle_depth"].sum(), depth, allow_unused=True)
            self.assertIsNone(gradient)
            replay.finish()
            cva.group.config.detach_depth = False
            replay.begin()
            ep = forward(depth)
            gradient, = torch.autograd.grad(ep["grasp_cdf_pred_angle_depth"].sum(), depth)
            self.assertGreater(float(gradient.norm()), 0)
            replay.finish()
            replay.begin()
            with self.assertRaisesRegex(RuntimeError, "seed_xyz"):
                forward(depth, seed_xyz=xyz + .001)
        self.assertEqual(cva.group._make_view_conditioned_grid, original_grid)
        self.assertFalse(cva._forward_pre_hooks)


class TrainingContractTests(unittest.TestCase):
    def test_original_supervised_objective_is_preserved(self):
        args = training_args()
        checkpoint = {"geometry_depth_source": "pred", "seed_selection_mode": "image_fps",
                      "pose_depth_mode": "none", "use_fuse_depth": False}
        cfgs = configure_repository(args, checkpoint)
        native = load_cpu_module("_depth_test_losses", "models/loss_economicgrasp_depth_kview_transformer.py")
        gt = torch.tensor([[[[.3, .4], [.5, .6]]]])
        ep = {"depth_map_pred": (gt + .01).requires_grad_(True), "gt_depth_m": gt,
              "objectness_score": torch.randn(1, 2, 4), "objectness_label_tok": torch.ones(1, 4, dtype=torch.long),
              "graspness_score": torch.randn(1, 1, 4), "graspness_label_tok": torch.rand(1, 4),
              "token_valid_mask": torch.ones(1, 4, dtype=torch.bool),
              "view_score": torch.randn(1, 2, 8), "batch_grasp_view_graspness": torch.rand(1, 2, 8),
              "grasp_cdf_pred_angle_depth": torch.randn(1, 6, 2, 12, 4),
              "batch_grasp_cdf_bins_angle_depth": torch.randint(0, 7, (1, 2, 12, 4)),
              "batch_grasp_cdf_valid_mask": torch.ones(1, 2, 12, 4, dtype=torch.bool),
              "batch_grasp_cdf_thresholds": torch.arange(1, 7) * .2,
              "grasp_width_pred_angle_depth": torch.rand(1, 4, 2, 12),
              "batch_grasp_width_angle_depth": torch.rand(1, 2, 12, 4) * .06,
              "batch_grasp_width_valid_mask_angle_depth": torch.ones(1, 2, 12, 4, dtype=torch.bool)}
        with patch.dict(sys.modules, {"models": SimpleNamespace(loss_economicgrasp_depth_kview_transformer=native)}):
            task, _ = grasp_objective(dict(ep), cfgs)
        actual, _ = native.get_loss_cdf(dict(ep))
        combined = task + args.metric_depth_weight * metric_depth_loss(ep["depth_map_pred"], gt, PairConfig())
        torch.testing.assert_close(combined, actual)
        with self.assertRaisesRegex(ValueError, "pose_depth_mode"):
            configure_repository(training_args(pose_depth_mode="global_film"), checkpoint)

    def test_holdout_scenes_excluded_before_frame_capping(self):
        class Frames:
            scenename = ["scene_0000"] * 4 + ["scene_0090"] * 4 + ["scene_0099"] * 4
            frameid = list(range(4)) * 3
            def __len__(self):
                return len(self.frameid)
        base = Frames()
        train = select_frame_indices(base, "", 1, 3, excluded_scenes={90, 99})
        valid = select_frame_indices(base, "90,99", 1, 4)
        self.assertEqual(train, [0, 1, 3])
        self.assertFalse(set(train) & set(valid))
        self.assertEqual({base.scenename[i] for i in valid}, {"scene_0090", "scene_0099"})
        self.assertEqual(training_args().eval_split, "train")

    def test_resume_rejects_changes_to_control_or_frame_contract(self):
        args = training_args()
        saved = {"depth_geometry_contract_version": 1, "depth_geometry_args": vars(args).copy(),
                 "world_size": 2, "train_indices": [1, 2], "eval_indices": [3, 4]}
        training.checked_resume(saved, args, [1, 2], [3, 4], 2)
        with self.assertRaises(ValueError):
            training.checked_resume(saved, args, [1, 2], [3, 5], 2)
        args.variant = "foreground"
        with self.assertRaises(ValueError):
            training.checked_resume(saved, args, [1, 2], [3, 4], 2)

    def test_detached_grasp_gradient_and_training_scopes(self):
        batch = synthetic_batch()

        class TinyDepth(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.depthnet = torch.nn.Module()
                self.depthnet.pretrained = torch.nn.Linear(1, 1).requires_grad_(False)
                self.scale = torch.nn.Parameter(torch.tensor(.8))
            def forward(self, img, **kwargs):
                return (img[:, :1] * self.scale,)

        class TinyStudent(torch.nn.Module):
            camera_pose_key, camera_gravity_key = "camera_pose_vec", "camera_gravity_vec"
            def __init__(self):
                super().__init__()
                self.depth_net = TinyDepth()
                self.grasp_weight = torch.nn.Parameter(torch.tensor(1.))
                self.view = torch.nn.Identity()
            def forward(self, ep):
                depth = self.depth_net(ep["img"])[0]
                ep.update(depth_map_pred=depth, task_loss=(depth.detach().mean() * self.grasp_weight).square())
                return ep

        with patch.object(training, "grasp_objective", lambda ep, cfgs: (ep["task_loss"], {})):
            for scope in ("joint", "depth_only"):
                model = TinyStudent()
                module = training.DepthTrainingModule(model, training_args(train_scope=scope), None)
                module.set_training(True)
                result = module(batch, pair_seed=42)
                if scope == "joint":
                    task_depth, = torch.autograd.grad(result["task"], model.depth_net.scale,
                                                     retain_graph=True, allow_unused=True)
                    self.assertIsNone(task_depth)
                result["loss"].backward()
                self.assertGreater(float(model.depth_net.scale.grad.abs()), 0)
                self.assertEqual(model.grasp_weight.grad is not None, scope == "joint")
                self.assertIsNone(model.depth_net.depthnet.pretrained.weight.grad)
                module.set_training(False)
                self.assertFalse(model.is_training)
                self.assertFalse(model.view.is_training)
                self.assertFalse(model.depth_net.training)

    def test_checkpoint_preserves_pose_dimensions(self):
        args = training_args(pose_depth_mode="global_film", use_fuse_depth=0)
        model = SimpleNamespace(camera_pose_key="camera_pose_vec", camera_gravity_key="camera_gravity_vec",
                                pose_hidden_dim=96, ray_gravity_hidden_dim=80, ray_gravity_mid_dim=40)
        metadata = checkpoint_metadata(model, args)
        self.assertEqual(metadata["pose_hidden_dim"], 96)
        self.assertEqual(metadata["geometry_depth_source"], "pred")

    def test_paired_summary_uses_each_frames_own_baseline(self):
        rows = []
        for idx, base in ((1, 1.), (2, 100.)):
            for beta, delta in ((1, 0), (0, -0.2)):
                rows.append({"protocol": "dynamic", "dataset_idx": idx, "scene_idx": idx,
                             "beta": beta, "grasp_total_loss": base + delta})
        summary = summarize(rows)
        self.assertAlmostEqual(summary[0]["grasp_total_loss_paired_delta_vs_beta1"], -.2)

    def test_help_does_not_import_cuda_stack(self):
        for name in ("train_cva_depth_geometry.py", "diagnose_cva_depth_contrast.py"):
            run = subprocess.run([sys.executable, str(ROOT / name), "--help"],
                                  cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("--checkpoint_path", run.stdout)


class BashLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = os.environ.get("DEPTH_TEST_BASH") or shutil.which("bash")
        if not cls.bash:
            raise unittest.SkipTest("Bash unavailable; run these tests in the Linux training environment.")

    def launch(self, script, **extra_env):
        env = dict(os.environ, DATASET_ROOT="/fixture/data with spaces", CHECKPOINT="/fixture/student with spaces.tar",
                   OUTPUT_ROOT="/fixture/output with spaces", DRY_RUN="1", RUN_TAG="selftest",
                   DEPTH_TEST_BASH=self.bash.replace("\\", "/"), NPROC_PER_NODE="1", RESUME="0",
                   VARIANT="anchor", PROBE_GEOMETRY_GRADIENT="0")
        env.pop("OUTPUT_DIR", None)
        env.update(extra_env)
        env["PATH"] = str(Path(self.bash).parent) + os.pathsep + env.get("PATH", "")
        # The bundled Windows GNU Bash is named sh.exe; allow child launcher
        # calls to use that same tested binary. Normal Linux uses bash directly.
        command = 'bash() { "$DEPTH_TEST_BASH" "$@"; }\nexport -f bash\nbash "$@"'
        run = subprocess.run([self.bash, "-c", command, "depth-launcher-test", f"scripts/{script}"],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        return [shlex.split(line.removeprefix("Running:")) for line in run.stdout.splitlines() if line.startswith("Running:")]

    def test_bash_syntax_and_lf(self):
        for name in ("depth_geometry_common.sh", "diagnose_cva_depth_contrast.sh", "train_cva_depth_geometry.sh",
                     "run_cva_depth_controls.sh"):
            path = ROOT / "scripts" / name
            self.assertNotIn(b"\r\n", path.read_bytes())
            run = subprocess.run([self.bash, "-n", str(path)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_diagnostic_and_ddp_resume_launch_arguments(self):
        diag, = self.launch("diagnose_cva_depth_contrast.sh", PROBE_GEOMETRY_GRADIENT="1")
        self.assertIn("--probe_geometry_gradient", diag)
        train, = self.launch("train_cva_depth_geometry.sh", NPROC_PER_NODE="2", RESUME="1", EVAL_SCENE_IDS="98,99")
        for token in ("torch.distributed.run", "--resume", "98,99"):
            self.assertIn(token, train)
        self.assertEqual(train[train.index("--dataset_root") + 1], "/fixture/data with spaces")
        self.assertEqual(train[train.index("--checkpoint_path") + 1], "/fixture/student with spaces.tar")

    def test_three_controls_have_same_initialization_and_distinct_outputs(self):
        commands = self.launch("run_cva_depth_controls.sh")
        self.assertEqual(len(commands), 3)
        values = lambda flag: [command[command.index(flag) + 1] for command in commands]
        self.assertEqual(values("--variant"), ["none", "foreground", "anchor"])
        self.assertEqual(len(set(values("--checkpoint_path"))), 1)
        self.assertEqual(len(set(values("--output_dir"))), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
