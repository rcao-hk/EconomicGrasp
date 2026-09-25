import torch

from dcr_air_common import (
    action_keypoints_camera, make_air_outputs, project_keypoints,
)
from models.economicgrasp_cva_air import ActionImageEvidenceReader


def _actions(c=1, q=1):
    a = torch.zeros(c, q, 17)
    a[..., 0] = .7
    a[..., 1] = .08
    a[..., 2] = .02
    a[..., 3] = .04
    a[..., 4:13] = torch.eye(3).reshape(1, 1, 9)
    a[..., 15] = .5
    a[..., 16] = -1
    return a


def test_action_keypoint_projection_identity_rotation():
    a = _actions()
    points = action_keypoints_camera(a)
    torch.testing.assert_close(points[0, 0, 0], a[0, 0, 13:16])
    # Identity R: inner contact points differ only in local +/- y.
    assert points[0, 0, 3, 1] < points[0, 0, 4, 1]
    K = torch.tensor([[[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]]])
    uv, visible = project_keypoints(points, K, (100, 100))
    torch.testing.assert_close(uv[0, 0, 0], torch.tensor([50., 50.]))
    assert bool(visible[0, 0, 0])


def test_air_reader_is_exact_zero_residual_at_initialization():
    torch.manual_seed(0)
    reader = ActionImageEvidenceReader(feature_dim=8, hidden=16, bound=1.)
    feat = torch.randn(1, 8, 25, 25)
    a = _actions(c=3, q=4)
    a[:, :, 13] = torch.linspace(-.05, .05, 4)[None]
    valid = torch.ones(3, 4, dtype=torch.bool)
    K = torch.tensor([[[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]]])
    residual, diag = reader(feat, K, (100, 100), a, valid)
    assert torch.count_nonzero(residual) == 0
    assert residual.shape == valid.shape
    assert diag['visible_ratio'].shape == valid.shape
    assert torch.isfinite(diag['attention_max']).all()


def test_air_output_keeps_stage1_score_and_only_changes_center():
    actions = _actions(c=3, q=2)
    actions[:, :, 15] = torch.tensor([[.48, .48], [.50, .50], [.52, .52]])
    actions[1, :, 0] = torch.tensor([.8, .6])
    valid = torch.ones(3, 2, dtype=torch.bool)
    bundle = {'actions': actions, 'valid': valid}

    base = torch.zeros(3, 2, 6)
    base[1] = 2.
    fused = base.clone()
    outputs, base_sel, air_sel = make_air_outputs(base, fused, bundle, zero=1)
    assert torch.equal(base_sel, air_sel)
    torch.testing.assert_close(outputs['dcr_stage1'], outputs['air_stage1'])

    fused = base.clone()
    fused[2, 0] = 8.
    outputs, base_sel, air_sel = make_air_outputs(base, fused, bundle, zero=1)
    assert base_sel[0] == 1
    assert air_sel[0] == 2
    assert outputs['air_stage1'][0, 15] == actions[2, 0, 15]
    assert outputs['air_stage1'][0, 0] == actions[1, 0, 0]
