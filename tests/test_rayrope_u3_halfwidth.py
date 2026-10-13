"""CPU-only contracts for frozen U3 RayRoPE halfwidth inference diagnostics."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_halfwidth_controls_are_deterministic_without_global_rng_changes():
    m = load_script("diagnose_rayrope_u3")
    s = (torch.arange(72, dtype=torch.float32).reshape(2, 1, 6, 6) + 1) * .001
    d = torch.full_like(s, .5)
    d[0, 0, 0, 0] = 0.
    frames = [(100, 0), (101, 10)]
    torch.testing.assert_close(m.intervention(s, d, "point", frames, .007, 10),
                               torch.zeros_like(s))
    torch.testing.assert_close(m.intervention(s, d, "fixed20", frames, .007, 10),
                               torch.full_like(s, .02))
    torch.testing.assert_close(m.intervention(s, d, "constant_train", frames, .007, 10),
                               torch.full_like(s, .007))
    assert m.intervention(s, d, "learned", frames, .007, 10) is s
    rng = torch.get_rng_state().clone()
    shuffle = m.intervention(s, d, "shuffle", frames, .007, 10)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for i in range(2):
        eligible = (d[i] > 0).flatten()
        torch.testing.assert_close(shuffle[i].flatten()[eligible].sort().values,
                                   s[i].flatten()[eligible].sort().values, rtol=0, atol=0)
    torch.testing.assert_close(shuffle,
        m.intervention(s, d, "shuffle", frames, .007, 10), rtol=0, atol=0)
    sep = torch.cat([m.intervention(s[i:i+1], d[i:i+1], "shuffle", [frames[i]], .007, 10)
                     for i in range(2)])
    torch.testing.assert_close(shuffle, sep, rtol=0, atol=0)


def test_group_hook_changes_only_halfwidth_and_keeps_upstream_proposals():
    m = load_script("diagnose_rayrope_u3")
    s = torch.tensor([[[[.003, .006], [.009, .012]]]])
    depth = torch.full_like(s, .5)
    class Group:
        geometry_context = {"sigma": s, "shape": None}
    kwargs = dict(
        depth_map=depth, seed_xyz=torch.randn(1, 4, 3),
        token_sel_idx=torch.arange(4).view(1, 4),
        top_view_rot=torch.eye(3).view(1, 1, 3, 3).expand(1, 4, 3, 3))
    c = m.Hook("point", .007, 10)
    c.before_model(None, (dict(scene_idx=torch.tensor([100]),
                               anno_idx=torch.tensor([10])),))
    c.before_group(Group, (), kwargs)
    assert Group.geometry_context["shape"] is None
    torch.testing.assert_close(Group.geometry_context["sigma"], torch.zeros_like(s))
    torch.testing.assert_close(s, torch.tensor([[[[.003, .006], [.009, .012]]]]))
    c.after_model(None, (), None)
    assert len(c.completed[(100, 10)]) == 64
    assert c.stats()["applied_halfwidth_mean_m"] == 0.


def test_compare_rejects_confounded_protocols():
    m = load_script("compare_rayrope_u3_halfwidth")
    a = dict(checkpoint_sha256="same", code_sha256="src", training_protocol={},
             checkpoint_epoch=19, split="test_novel", schedule=[[160, 0]],
             batch_size=3, num_shards=3, collision_filter="none",
             depth_bias_mm=0., dataset_root="test", evaluation_fraction=.1, seed=0,
             diagnostic=dict(script_sha256="d", train_metrics_sha256="m",
                             constant_train_halfwidth_m=.0077, shuffle_seed=20261013))
    b = dict(a)
    m._same_contract(a, b)
    b = dict(a, batch_size=1)
    with pytest.raises(ValueError, match="batch_size"):
        m._same_contract(a, b)
    b = dict(a, checkpoint_sha256="other")
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        m._same_contract(a, b)


def test_nonfinite_sigma_or_batch_mismatch_rejected():
    m = load_script("diagnose_rayrope_u3")
    sigma = torch.ones(2, 1, 3, 3)
    depth = torch.ones_like(sigma)
    with pytest.raises(RuntimeError):
        m.intervention(sigma, depth, "point", [(0, 0)], .007, 1)
    sigma[0, 0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError):
        m.intervention(sigma, depth, "learned", [(0, 0), (1, 0)], .007, 1)
