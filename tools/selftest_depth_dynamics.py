"""CPU invariants for diagnostics (no project datasets or compiled ops needed).

Run: python tools/selftest_depth_dynamics.py
This validates utilities; real-model P0 forward/connectivity gates remain required.
"""
from __future__ import annotations

import copy
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depth_dynamics import (  # noqa: E402
    append_csv, append_jsonl, audit_gradients, capture_rng_state, depth_metrics,
    make_fixed_pairs, output_directional_derivatives, preserve_diagnostic_state,
    restore_rng_state, restore_snapshot, sha256_file, snapshot_state, stable_hash,
    write_json,
)


def assert_tree_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b), (a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b, (a, b)


def test_gradient_states_and_weights():
    p = nn.Parameter(torch.tensor([1., -2.]))
    zero = nn.Parameter(torch.tensor([3.]))
    unused = nn.Parameter(torch.tensor([4.]))
    frozen = nn.Parameter(torch.tensor([5.]), requires_grad=False)
    p.grad = torch.tensor([9., 8.])
    grad_before = p.grad.clone()
    losses = {"depth": p.square().sum() + 0 * zero.sum(),
              "task": -p.square().sum() + 0 * zero.sum(),
              "detached": torch.tensor(2.)}
    groups = {"active": [("p", p)], "zero": [("zero", zero)],
              "unused": [("unused", unused)], "frozen": [("frozen", frozen)]}
    rows = audit_gradients(losses, groups, {"depth": 2., "task": .5})
    by_key = {(r["loss"], r["scale"], r["group"]): r for r in rows}
    assert by_key["depth", "raw", "active"]["state"] == "connected_nonzero"
    assert by_key["depth", "raw", "zero"]["state"] == "connected_zero"
    assert by_key["depth", "raw", "unused"]["state"] == "unused/None"
    assert by_key["depth", "raw", "frozen"]["state"] == "not_requires_grad"
    assert by_key["depth", "raw", "zero"]["connected_numel"] == 1
    assert by_key["depth", "raw", "unused"]["connected_numel"] == 0
    assert by_key["depth", "raw", "zero"]["cosine_vs_depth"] is None
    assert by_key["detached", "raw", "active"]["norm"] is None
    task = by_key["task", "weighted", "active"]
    assert abs(task["cosine_vs_depth"] + 1) < 1e-12
    assert abs(task["norm_ratio_vs_depth"] - .25) < 1e-12
    assert torch.equal(p.grad, grad_before)
    losses["depth"].backward()
    assert torch.equal(p.grad, grad_before + 2 * p.detach())
    nanloss = (p * float("nan")).sum()
    nanrows = audit_gradients({"depth": p.sum(), "bad": nanloss}, {"active": [("p", p)]})
    assert next(r for r in nanrows if r["loss"] == "bad")["state"] == "nonfinite"
    assert next(r for r in nanrows if r["loss"] == "bad")["norm"] is None


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        self.drop = nn.Dropout(.25)
        self.linear = nn.Linear(3, 1)
        self.is_training = True
        self._vis_iter = 2

    def forward(self, x):
        self._vis_iter += 1
        return self.linear(self.drop(self.bn(x)))


def test_state_guard_and_trajectory():
    random.seed(3)
    np.random.seed(4)
    torch.manual_seed(5)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    x = torch.randn(8, 3)

    def step():
        optimizer.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        optimizer.step()

    step()  # Nonempty Adam state and .grad are important to this invariant.
    initial = snapshot_state(model, optimizer, step=1, loader_position=8)
    initial_grads = [(p.grad, p.grad.clone()) for p in model.parameters()]
    before_rng = capture_rng_state()
    try:
        with preserve_diagnostic_state(model, optimizer):
            model.eval()
            model.bn.train()  # Deliberately create a mixed-mode probe.
            model.is_training = False
            model._vis_iter = 99
            random.random()
            np.random.rand()
            output = model(x)
            audit_gradients({"depth": output.square().mean(), "task": output.sum()},
                            {"all": list(model.named_parameters())})
            for p in model.parameters():
                p.grad = None
            optimizer.param_groups[0]["lr"] = 9.
            raise RuntimeError("probe failed deliberately")
    except RuntimeError as error:
        assert str(error) == "probe failed deliberately"
    assert_tree_equal(capture_rng_state(), before_rng)
    assert_tree_equal(snapshot_state(model, optimizer, step=1, loader_position=8), initial)
    for p, (original, value) in zip(model.parameters(), initial_grads):
        assert p.grad is original and torch.equal(p.grad, value)

    step()
    expected = snapshot_state(model, optimizer)
    restore_snapshot(initial, model, optimizer)
    with preserve_diagnostic_state(model, optimizer):
        output = model(x)
        audit_gradients({"depth": output.square().mean(), "task": output.sum()},
                        {"all": list(model.named_parameters())})
    step()
    actual = snapshot_state(model, optimizer)
    assert_tree_equal(actual, expected)


def test_fixed_metrics_and_rng():
    gt = torch.linspace(.25, .55, 16).reshape(1, 4, 4).repeat(2, 1, 1)
    instance = torch.ones_like(gt, dtype=torch.long)
    instance[:, :, 2:] = 2
    rng = capture_rng_state()
    pairs = make_fixed_pairs(gt, instance, max_pairs=9, seed=7)
    assert_tree_equal(capture_rng_state(), rng)
    assert_tree_equal(pairs, make_fixed_pairs(gt, instance, max_pairs=9, seed=7))
    for i, pair in enumerate(pairs):
        ids = instance[i].flatten()[pair]
        assert bool((ids[:, 0] == ids[:, 1]).all())
    pred = 2 * gt + .1
    raw = torch.zeros_like(gt)
    rows = depth_metrics(pred, gt, foreground=instance == 1, raw=raw, pairs=pairs)
    for row in rows:
        assert abs(row["local"]["slope"] - 2) < 1e-5
        assert abs(row["local"]["contrast_ratio"] - 2) < 1e-5
        assert abs(row["local"]["correlation"] - 1) < 1e-5
        assert row["raw"]["sigmoid_derivative_mean"] == .25
        assert row["regions"]["foreground"]["count"] == 8
    gt_flat = torch.stack((torch.full((4, 4), .3), torch.full((4, 4), .8)))
    flat_rows = depth_metrics(gt_flat, gt_flat, pairs=make_fixed_pairs(gt_flat))
    assert all(row["regions"]["valid"]["pred_std"] < 1e-7 for row in flat_rows)
    assert all(row["local"]["contrast_ratio"] is None for row in flat_rows)
    invalid = depth_metrics(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2),
                            pairs=[torch.empty((0, 2), dtype=torch.long)])
    assert invalid[0]["regions"]["valid"]["mae"] is None
    assert invalid[0]["local"]["slope"] is None
    pred[0, 0, 0] = float("nan")
    row = depth_metrics(pred, gt)[0]
    assert row["pred_nonfinite_count"] == 1
    assert row["regions"]["valid"]["count"] == 16
    assert row["regions"]["valid"]["finite_count"] == 15


def test_repeated_adam_snapshot_replay():
    torch.manual_seed(19)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    x, target = torch.randn(5, 3), torch.randn(5, 2)

    def step():
        optimizer.zero_grad(set_to_none=True)
        (model(x)-target).square().mean().backward()
        optimizer.step()
        scheduler.step()

    step()  # Populate moments and Adam's mutable CPU step tensors.
    reference = snapshot_state(model, optimizer, scheduler, step=1)
    pristine_reference = copy.deepcopy(reference)
    outcomes = []
    for _ in range(3):
        restore_snapshot(reference, model, optimizer, scheduler)
        step()
        outcomes.append(snapshot_state(model, optimizer, scheduler, step=2))
        assert_tree_equal(reference, pristine_reference)
    for outcome in outcomes[1:]:
        assert_tree_equal(outcome, outcomes[0])


def test_directional_derivative():
    pred = torch.tensor([[[.3, .4], [.6, .8]]], dtype=torch.float64, requires_grad=True)
    gt = torch.tensor([[[.4, .5], [.7, .9]]], dtype=torch.float64)
    objective = lambda z: ((z-gt).square()).mean()
    row = output_directional_derivatives({"depth": objective(pred)}, pred, gt)[0]
    mu0 = pred.detach().mean()
    contrast = pred.detach()-mu0
    epsilon = 1e-5
    dmu = (objective(pred.detach()+epsilon)-objective(pred.detach()-epsilon))/(2*epsilon)
    dalpha = (objective(mu0+(1+epsilon)*contrast)-objective(mu0+(1-epsilon)*contrast))/(2*epsilon)
    assert abs(row["d_loss_d_mu"]-float(dmu)) < 1e-10
    assert abs(row["d_loss_d_alpha"]-float(dalpha)) < 1e-10


def test_serialization():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        data = {"finite": torch.tensor([1., 2.]), "undefined": float("nan")}
        write_json(root / "contract.json", data)
        assert json.loads((root / "contract.json").read_text())["undefined"] is None
        assert len(sha256_file(root / "contract.json")) == 64
        assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
        append_jsonl(root / "audit.jsonl", data)
        append_csv(root / "audit.csv", [{"loss": "depth", "norm": 1.}])
        append_csv(root / "audit.csv", [{"loss": "task", "norm": None}])
        try:
            append_csv(root / "audit.csv", [{"wrong": 1}])
            raise AssertionError("CSV schema drift was not rejected")
        except ValueError:
            pass


if __name__ == "__main__":
    tests = [test_gradient_states_and_weights, test_state_guard_and_trajectory,
             test_fixed_metrics_and_rng, test_repeated_adam_snapshot_replay,
             test_directional_derivative, test_serialization]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"PASS {len(tests)} depth-dynamics utility invariant tests")
