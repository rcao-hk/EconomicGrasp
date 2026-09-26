import torch

from rep_p1_common import (
    REGION_POINT_COUNTS,
    REGION_POINTS,
    ActionPointImageProbe,
    ActionRegionImageProbe,
    action_point_keypoints_camera,
    action_region_points_camera,
    pairwise_rank_loss,
    project_points,
)


def _actions(k=3, q=4):
    a = torch.zeros(k, q, 17)
    a[..., 0] = 0.7
    a[..., 1] = 0.08
    a[..., 2] = 0.02
    a[..., 3] = 0.04
    a[..., 4:13] = torch.eye(3).reshape(1, 1, 9)
    a[..., 15] = 0.5
    a[..., 16] = -1
    for i in range(k):
        a[i, :, 15] += (i - k // 2) * 0.01
    return a


def _camera():
    return torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )


def test_sparse_keypoints_are_action_aligned():
    a = _actions(k=1, q=1)
    pts = action_point_keypoints_camera(a)
    assert pts.shape == (1, 1, 13, 3)
    torch.testing.assert_close(pts[0, 0, 0], a[0, 0, 13:16])
    uv, visible = project_points(pts, _camera(), (100, 100))
    assert uv.shape == (1, 1, 13, 2)
    assert bool(visible[0, 0, 0])


def test_region_reader_uses_declared_structured_regions():
    a = _actions(k=1, q=1)
    pts, slot, rid = action_region_points_camera(a)
    assert pts.shape == (1, 1, REGION_POINTS, 3)
    assert slot.shape == (REGION_POINTS, 3)
    assert rid.shape == (REGION_POINTS,)
    counts = [int((rid == i).sum()) for i in range(len(REGION_POINT_COUNTS))]
    assert tuple(counts) == tuple(REGION_POINT_COUNTS)


def test_image_probes_predict_six_threshold_logits_and_backpropagate():
    torch.manual_seed(0)
    a = _actions()
    valid = torch.ones(a.shape[:2], dtype=torch.bool)
    feature = torch.randn(1, 8, 25, 25)
    K = _camera()

    for cls in (ActionPointImageProbe, ActionRegionImageProbe):
        model = cls(feature_dim=8, hidden=16, dropout=0.0)
        logits, diag = model(feature, K, (100, 100), a, valid)
        assert logits.shape == (*a.shape[:2], 6)
        assert torch.isfinite(logits).all()
        assert "visible_ratio" in diag
        loss = logits.square().mean()
        loss.backward()
        grad = sum(
            float(p.grad.abs().sum())
            for p in model.parameters()
            if p.grad is not None
        )
        assert grad > 0.0


def test_pairwise_rank_loss_prefers_correct_within_ray_order():
    # K=3, Q=1.  Exact utility is increasing with k.
    exact = torch.tensor([[0.0], [0.5], [1.0]])
    valid = torch.ones(3, 1, dtype=torch.bool)

    good = torch.tensor([
        [[-6.0] * 6],
        [[0.0] * 6],
        [[6.0] * 6],
    ])
    bad = good.flip(0)
    loss_good = pairwise_rank_loss(good, exact, valid, temperature=0.1)
    loss_bad = pairwise_rank_loss(bad, exact, valid, temperature=0.1)
    assert float(loss_good) < float(loss_bad)
