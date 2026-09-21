"""Rep-B CPU tests: invariance, prior sensitivity, masking and gradients."""
import numpy as np
import pytest
import torch

from rep_b_common import cdf_bce_loss, pairwise_ranking_loss
from rep_b_model import RepBModel


def synthetic(k=3, q=4, channels=8, h=32, w=32):
    a = torch.zeros(k, q, 17)
    a[..., 0] = .6
    a[..., 1] = .06
    a[..., 2] = .02
    a[..., 3] = .03
    a[..., 4:13] = torch.eye(3).reshape(9)
    z = torch.tensor([.48, .50, .52])[:k]
    a[..., 15] = z[:, None]
    # camera x/y are small enough that projected gripper volumes remain visible
    a[..., 13] = torch.linspace(-.01, .01, q)[None]
    valid = torch.ones(k, q, dtype=torch.bool)
    d = {
        "image_feature": torch.randn(channels, 8, 8),
        "depth": torch.ones(1, h, w) * .50,
        "K": torch.tensor([[60., 0., 15.5], [0., 60., 15.5], [0., 0., 1.]]),
        "actions": a,
        "valid": valid,
        "zero_index": 1,
        "offsets_mm": torch.tensor([-20., 0., 20.])[:k],
        "token_ids": torch.tensor([15*w+14, 15*w+15, 15*w+16, 15*w+17])[:q],
    }
    friction = np.array([
        [.8, .6, .4, .8],
        [.6, .4, .8, .6],
        [.4, .8, .6, .4],
    ], dtype=np.float32)[:k, :q]
    utility = np.array([
        [.2, .4, .6, .2],
        [.4, .6, .2, .4],
        [.6, .2, .4, .6],
    ], dtype=np.float32)[:k, :q]
    return d, friction, utility


def make(variant, seed=0):
    torch.manual_seed(seed)
    return RepBModel(
        channels=8, dim=32, heads=4, layers=1, dropout=0.,
        prior_sigma_mm=30., variant=variant,
    ).eval()


def test_b0_is_depth_invariant():
    d, _, _ = synthetic()
    m = make("B0")
    with torch.no_grad():
        a, ra = m(d, return_repr=True)
        changed = d["depth"] + .04
        b, rb = m(d, depth=changed, return_repr=True)
    assert torch.equal(a, b)
    assert torch.equal(ra, rb)


@pytest.mark.parametrize("variant", ["B1", "B2"])
def test_soft_prior_responds_to_depth(variant):
    d, _, _ = synthetic()
    m = make(variant)
    with torch.no_grad():
        a, ra, ca = m(d, return_repr=True, return_components=True)
        b, rb, cb = m(d, depth=d["depth"] + .04,
                      return_repr=True, return_components=True)
    assert not torch.allclose(ca["prior"], cb["prior"])
    assert not torch.allclose(ra, rb)
    assert not torch.allclose(a, b)
    # RGB image evidence itself is independent of the depth observation.
    assert torch.equal(ca["image"], cb["image"])


def test_hypotheses_receive_distinct_rgb_action_tokens():
    d, _, _ = synthetic()
    m = make("B0")
    with torch.no_grad():
        _, _, c = m(d, return_repr=True, return_components=True)
    # same ray/query, different physical z hypotheses
    assert not torch.allclose(c["image"][0, 0], c["image"][2, 0])
    assert not torch.allclose(c["action"][0, 0], c["action"][2, 0])
    assert not torch.allclose(c["offset"][0, 0], c["offset"][2, 0])


def test_invalid_nan_hypothesis_is_masked_safely():
    d, friction, utility = synthetic()
    d["valid"][0, 0] = False
    d["actions"][0, 0] = float("nan")
    m = make("B1")
    logits = m(d)
    assert torch.isfinite(logits).all()
    loss = cdf_bce_loss(logits, friction, d["valid"])
    rank = pairwise_ranking_loss(logits, utility, d["valid"])
    assert torch.isfinite(loss)
    assert torch.isfinite(rank)


@pytest.mark.parametrize("variant", ["B0", "B1", "B2"])
def test_all_trainable_rep_b_components_get_gradient(variant):
    d, friction, utility = synthetic()
    torch.manual_seed(3)
    m = RepBModel(
        channels=8, dim=32, heads=4, layers=1, dropout=.1,
        prior_sigma_mm=30., variant=variant,
    ).train()
    logits = m(d)
    loss = cdf_bce_loss(logits, friction, d["valid"])
    loss.backward()
    grads = {}
    for name, part in m.named_children():
        params = [p for p in part.parameters() if p.requires_grad]
        if params:
            grads[name] = sum(
                float(p.grad.norm()) for p in params if p.grad is not None
            )
    assert grads
    assert all(np.isfinite(v) and v > 0 for v in grads.values()), grads


def test_common_initialization_is_paired_across_variants():
    b0 = make("B0", seed=11)
    b1 = make("B1", seed=11)
    common = ("image_reader", "offset_embed", "action_embed", "relational", "scorer")
    for name in common:
        a = getattr(b0, name).state_dict()
        b = getattr(b1, name).state_dict()
        assert a.keys() == b.keys()
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_pairwise_loss_prefers_correct_order():
    _, _, utility = synthetic()
    valid = torch.ones(3, 4, dtype=torch.bool)
    # construct six identical CDF logits whose mean follows utility ordering
    u = torch.tensor(utility)
    p_good = (.05 + .9*u).clamp(.01, .99)
    p_bad = 1-p_good
    lg = torch.logit(p_good)[..., None].expand(-1, -1, 6).clone()
    lb = torch.logit(p_bad)[..., None].expand(-1, -1, 6).clone()
    good = pairwise_ranking_loss(lg, utility, valid, temperature=.1)
    bad = pairwise_ranking_loss(lb, utility, valid, temperature=.1)
    assert good < bad
