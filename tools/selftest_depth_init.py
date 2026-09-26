"""Small cold-init/schedule/gradient-policy checks; torch tests run in project env."""
from pathlib import Path
from types import SimpleNamespace
import copy
import importlib.util
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import depth_init
import train_cva_depth_dynamics as driver


class InitializationContract(unittest.TestCase):
    def test_probe_schedule(self):
        args = SimpleNamespace(probe_schedule="cold", probe_interval=100)
        actual = [s for s in range(1101) if driver.probe_due(s, args)]
        self.assertEqual(actual, [0, 10, 25, 50, 100, *range(150, 1001, 50), 1100])

    def test_cold_requires_shared_canonical(self):
        with self.assertRaises(SystemExit):
            driver.parse_args(["--output", "unused", "--init_mode", "current_standard_cold",
                               "--architecture_checkpoint", "metadata.pt"])

    def test_policy_requires_open_routes(self):
        with self.assertRaises(SystemExit):
            driver.parse_args(["--output", "unused", "--init_checkpoint", "warm.pt",
                               "--depth_gradient_policy", "remove_view_reclip"])

    def test_canonical_arguments(self):
        args, rest = driver.parse_args(["--output", "unused", "--init_mode", "current_standard_cold",
                                      "--canonical_init", "canonical.pt", "--probe_schedule", "cold"])
        self.assertEqual(args.seed, 0)
        self.assertEqual(rest, [])

    def test_resume_rejects_changed_gradient_policy(self):
        args = SimpleNamespace(init_mode="current_standard_cold", depth_gradient_policy="normal", probe_schedule="cold")
        state = {"arguments": dict(vars(args), depth_gradient_policy="remove_view_reclip")}
        with self.assertRaises(ValueError):
            depth_init.assert_resume_settings(state, args)


@unittest.skipUnless(importlib.util.find_spec("torch"), "project torch environment required")
class GradientPolicy(unittest.TestCase):
    def test_cold_source_does_not_load_task_weights(self):
        import torch
        model = torch.nn.Linear(2, 1)
        before = copy.deepcopy(model.state_dict())
        result = depth_init.apply_source(model, {"architecture": {}, "provenance": {}})
        self.assertEqual(result["loaded_keys"], [])
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))

    def test_selective_removal_matches_retained_objective_with_reclip(self):
        import torch
        p, q = torch.nn.Parameter(torch.tensor(.3)), torch.nn.Parameter(torch.tensor(.7))
        a, b = torch.nn.Parameter(p.detach().clone()), torch.nn.Parameter(q.detach().clone())
        opt1, opt2 = torch.optim.AdamW([p, q], lr=.001, weight_decay=0), torch.optim.AdamW([a, b], lr=.001, weight_decay=0)
        for _ in range(20):
            opt1.zero_grad(set_to_none=True); opt2.zero_grad(set_to_none=True)
            view = 7 * p * q
            depth_init.backward_with_policy(p.square() + 3*p.square() + 2*p.square() + view,
                view, [("depth", p)], "remove_view_reclip", [p, q])
            (a.square() + 3*a.square() + 2*a.square() + 7*a.detach()*b).backward()
            torch.nn.utils.clip_grad_norm_([p, q], 1.)
            torch.nn.utils.clip_grad_norm_([a, b], 1.)
            opt1.step(); opt2.step()
            self.assertTrue(torch.allclose(p, a, atol=1e-7, rtol=1e-6))
            self.assertTrue(torch.allclose(q, b, atol=1e-7, rtol=1e-6))

    def test_noop_preserves_rng_and_none_zero(self):
        import torch
        p, zero, unused = [torch.nn.Parameter(torch.tensor(v)) for v in (.3, .2, .8)]
        rng = torch.get_rng_state().clone()
        view = p.square()
        depth_init.backward_with_policy(view + p*3 + zero*0, view,
            [("depth", p), ("zero", zero), ("unused", unused)], "view_noop", [p, zero, unused])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIsNone(unused.grad)
        self.assertEqual(float(zero.grad), 0.)
        self.assertAlmostEqual(float(p.grad), 3.6, places=6)


if __name__ == "__main__":
    unittest.main()
