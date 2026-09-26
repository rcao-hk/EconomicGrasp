"""CPU-only counterfactual AdamW invariants; no model or dataset imports.

Run with CUDA_VISIBLE_DEVICES='' using an environment containing CPU-capable
PyTorch. The warmed optimizer is deliberately part of every branch's input.
"""
from __future__ import annotations

import copy
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import depth_dynamics as dd
import replay_cva_depth_counterfactual as replay

# The production CLI imports these lazily, after parsing its own arguments.
replay.torch = torch
replay.dd = dd


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = nn.Module()
        self.depth.register_parameter("scale", nn.Parameter(torch.tensor(.3, dtype=torch.float64)))
        self.depth.register_parameter("bias", nn.Parameter(torch.tensor(-.4, dtype=torch.float64)))
        self.shared = nn.Parameter(torch.tensor(.6, dtype=torch.float64))
        self.connected_zero = nn.Parameter(torch.tensor(-.2, dtype=torch.float64))
        self.unused = nn.Parameter(torch.tensor(.9, dtype=torch.float64))
        self.register_buffer("running", torch.tensor([1., 2.], dtype=torch.float64))
        self.is_training = True
        self.depth.is_training = False
        self._vis_iter = 7
        self.depth._debug_iter = 3
        self.routes = {"gse": True, "seed_xyz": False, "support": True}

    def get_depth_grad_routes(self):
        return dict(self.routes)

    def set_depth_grad_routes(self, routes):
        if isinstance(routes, dict):
            self.routes = dict(routes)
        else:
            enabled = {"gse", "seed_xyz", "support"} if routes == "all" else set(routes.split(","))
            self.routes = {name: name in enabled for name in self.routes}


class ToyStream:
    def __init__(self):
        self.value = {"epoch": 2, "offset": 11, "seed": 901}

    def state_dict(self):
        return copy.deepcopy(self.value)

    def load_state_dict(self, state):
        self.value = copy.deepcopy(state)


def full_state(exp):
    return copy.deepcopy({
        "model_state_dict": exp.model.state_dict(),
        "optimizer_state_dict": exp.optimizer.state_dict(),
        "loader": exp.stream.state_dict(), "rng": dd.capture_rng_state(),
        "step": exp.step, "seen_images": exp.seen_images,
        "module_training": {name: module.training for name, module in exp.model.named_modules()},
        "is_training": {name: module.is_training for name, module in exp.model.named_modules()
                        if hasattr(module, "is_training")},
        "runtime_counters": {name: {key: getattr(module, key) for key in ("_vis_iter", "_debug_iter")
                                    if hasattr(module, key)} for name, module in exp.model.named_modules()},
        "routes": exp.model.get_depth_grad_routes(),
    })


class CounterfactualStepTest(unittest.TestCase):
    def assertTreeEqual(self, actual, expected):
        if torch.is_tensor(expected):
            self.assertTrue(torch.is_tensor(actual))
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(torch.equal(actual, expected), f"{actual} != {expected}")
        elif isinstance(expected, np.ndarray):
            self.assertTrue(np.array_equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assertTreeEqual(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assertTreeEqual(left, right)
        else:
            self.assertEqual(actual, expected)

    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model = ToyModel()
        model.train()
        model.depth.eval()  # Mixed module modes must survive restore.
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, betas=(.8, .9),
                                      eps=1e-8, weight_decay=.1, foreach=False)
        self.exp = SimpleNamespace(model=model, optimizer=optimizer, stream=ToyStream(),
                                   step=12, seen_images=24)
        # A nonzero prehistory makes zero-gradient and missing-gradient updates
        # observably different, and exposes accidental Adam state aliasing.
        for index in range(3):
            optimizer.zero_grad(set_to_none=True)
            warm = ((index + 1.) * model.depth.scale - .4 * model.depth.bias
                    + .6 * model.shared + .7 * model.connected_zero)
            warm.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.state = full_state(self.exp)
        self.state_original = copy.deepcopy(self.state)
        named = dict(model.named_parameters())
        task = 6. * model.depth.scale - 4. * model.depth.bias + 3. * model.shared
        depth = 1.25 * model.depth.scale - .5 * model.depth.bias
        other = .75 * model.shared + 0. * model.connected_zero
        total = depth + .7 * task + other
        all_grads = torch.autograd.grad(total, tuple(named.values()), retain_graph=True, allow_unused=True)
        depth_names = ("depth.scale", "depth.bias")
        removed = torch.autograd.grad(.7 * task, tuple(named[name] for name in depth_names))
        self.total = {name: None if value is None else value.detach().clone()
                      for name, value in zip(named, all_grads)}
        self.removed = {name: value.detach().clone() for name, value in zip(depth_names, removed)}

    def branch(self, removed=None, fixed_coefficient=None):
        replay.restore_full_state(self.exp, self.state)
        replay.install_gradients(self.exp.model, self.total, removed)
        before_clip = {name: None if p.grad is None else p.grad.clone()
                       for name, p in self.exp.model.named_parameters()}
        norm, coefficient = replay.apply_clip(self.exp.model, 1., fixed_coefficient=fixed_coefficient)
        self.exp.optimizer.step()
        parameters = {name: p.detach().clone() for name, p in self.exp.model.named_parameters()}
        return {"parameters": parameters, "gradients": before_clip, "norm": float(norm),
                "coefficient": float(coefficient),
                "optimizer": copy.deepcopy(self.exp.optimizer.state_dict())}

    def test_four_branch_adamw_replay_and_scoped_removal(self):
        normal_a = self.branch()
        normal_b = self.branch()
        removed_fixed = self.branch(self.removed, normal_a["coefficient"])
        removed_recomputed = self.branch(self.removed)
        self.assertTreeEqual(normal_a, normal_b)
        self.assertLess(normal_a["coefficient"], 1.)
        self.assertGreater(removed_recomputed["coefficient"], normal_a["coefficient"])
        for name in self.total:
            if name not in self.removed:
                self.assertTreeEqual(removed_fixed["gradients"][name], normal_a["gradients"][name])
                self.assertTreeEqual(removed_recomputed["gradients"][name], normal_a["gradients"][name])
                self.assertTreeEqual(removed_fixed["parameters"][name], normal_a["parameters"][name])
            else:
                self.assertTreeEqual(removed_fixed["gradients"][name], self.total[name] - self.removed[name])
                self.assertFalse(torch.equal(removed_fixed["parameters"][name], normal_a["parameters"][name]))
        # Recomputing the global clip coefficient is allowed to change shared
        # updates; the fixed-coefficient branch must isolate that effect.
        self.assertFalse(torch.equal(removed_recomputed["parameters"]["shared"],
                                     normal_a["parameters"]["shared"]))
        self.assertTreeEqual(self.state, self.state_original)

    def test_none_and_connected_zero_have_distinct_adamw_semantics(self):
        self.assertIsNone(self.total["unused"])
        self.assertEqual(self.total["connected_zero"].item(), 0.)
        branch = self.branch()
        before = self.state["model_state_dict"]
        self.assertTreeEqual(branch["parameters"]["unused"], before["unused"])
        self.assertFalse(torch.equal(branch["parameters"]["connected_zero"], before["connected_zero"]))
        self.assertIsNone(dict(self.exp.model.named_parameters())["unused"].grad)

    def test_installed_gradients_do_not_alias_capture(self):
        original = copy.deepcopy(self.total)
        replay.install_gradients(self.exp.model, self.total, self.removed)
        for parameter in self.exp.model.parameters():
            if parameter.grad is not None:
                parameter.grad.add_(123.)
        self.assertTreeEqual(self.total, original)

    def test_cannot_remove_connected_gradient_from_missing_total(self):
        with self.assertRaises((ValueError, RuntimeError)):
            replay.install_gradients(self.exp.model, self.total, {"unused": torch.tensor(0., dtype=torch.float64)})

    def test_removed_names_must_exist(self):
        with self.assertRaises((ValueError, RuntimeError, KeyError)):
            replay.install_gradients(self.exp.model, self.total, {"not_a_parameter": torch.tensor(1.)})

    def test_full_restore_after_interrupted_branch(self):
        expected_rng = self.state["rng"]
        try:
            self.branch(self.removed)
            self.exp.model.running.add_(5.)
            self.exp.model.eval()
            self.exp.model.depth.train()
            self.exp.model.is_training = False
            self.exp.model.depth.is_training = True
            self.exp.model._vis_iter = 999
            self.exp.model.depth._debug_iter = 999
            self.exp.model.set_depth_grad_routes("none")
            self.exp.stream.value["offset"] += 5
            self.exp.step += 1
            self.exp.seen_images += 2
            random.random(), np.random.rand(), torch.rand(2)
            raise RuntimeError("simulated branch failure")
        except RuntimeError as error:
            self.assertEqual(str(error), "simulated branch failure")
        finally:
            replay.restore_full_state(self.exp, self.state)
        self.assertTreeEqual(full_state(self.exp), self.state)
        self.assertTreeEqual(dd.capture_rng_state(), expected_rng)
        # A subsequent branch must behave exactly as a fresh replay, including
        # the warmed momentum and per-parameter Adam step counters.
        first = self.branch()
        second = self.branch()
        self.assertTreeEqual(first, second)
        self.assertTreeEqual(self.state, self.state_original)


if __name__ == "__main__":
    unittest.main()
