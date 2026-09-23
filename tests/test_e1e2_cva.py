"""CPU contract/gradient tests. Main Stage-1/CAD adapters need a server smoke."""
import copy
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from e1e2_common import (array_sha, cdf_targets, expand_centers, frame_stride,
                         perturb_depth, relative_cdf_loss, schedule, select_centers, shard_frames)
from models.economicgrasp_cva_centers import CenterHypothesisCVA, _depth_override


def test_ten_percent_is_a_frame_schedule_not_a_query_limit():
    assert frame_stride(.1) == 10
    assert len(schedule('train')) == 2600
    for split in ('test_seen', 'test_similar', 'test_novel'):
        assert len(schedule(split)) == 780
    assert [a for _, a in schedule('train')[:26]] == list(range(0, 256, 10))
    assert max(s for s, _ in schedule('test_seen')) == 129
    with pytest.raises(ValueError):
        frame_stride(10)


def test_sharding_reassigns_same_frames_and_limits_before_workers():
    frames = schedule('train')
    pieces = [set(shard_frames(frames, i, 4)) for i in range(4)]
    assert set.union(*pieces) == set(frames)
    assert sum(map(len, pieces)) == len(frames)
    small = [set(shard_frames(frames, i, 4, 30)) for i in range(4)]
    assert set.union(*small) == set(frames[:30])


def native_actions(q=4):
    x = torch.zeros(q, 17)
    x[:, 0] = .5; x[:, 1] = .06; x[:, 2] = .02; x[:, 3] = .01
    x[:, 4:13] = torch.eye(3).flatten(); x[:, 13] = .1; x[:, 14] = -.02; x[:, 15] = .5
    return x


def test_offsets_translate_camera_ray_without_changing_rwd():
    n = native_actions()
    a, v = expand_centers(n, [-40., 0., 20.])
    torch.testing.assert_close(a[1], n)
    torch.testing.assert_close(a[2, :, 15]-n[:, 15], torch.full((4,), .02))
    torch.testing.assert_close(a[:, :, 13]/a[:, :, 15], (n[:, 13]/n[:, 15]).expand(3, 4))
    torch.testing.assert_close(a[:, :, 1:13], n[None, :, 1:13].expand(3, 4, 12))
    assert v.all()


def test_cdf_targets_failure_and_monotonicity():
    y = cdf_targets(np.asarray([-1., .2, .6, 1.2]))
    np.testing.assert_array_equal(y[0], np.zeros(6))
    assert (np.diff(y, axis=-1) >= 0).all()
    np.testing.assert_array_equal(y[2], [0, 0, 1, 1, 1, 1])


def test_e2_changes_only_relative_objective_and_backpropagates():
    torch.manual_seed(2)
    logits = torch.randn(3, 4, 6, requires_grad=True)
    y = torch.tensor(cdf_targets(np.array([[.2,.4,.6,.8], [.4,.8,-1,.2], [.8,.4,.6,1.2]])))
    valid = torch.ones(3, 4, dtype=torch.bool)
    l1, m1 = relative_cdf_loss(logits, y, valid, 1, 'E1')
    l2, m2 = relative_cdf_loss(logits, y, valid, 1, 'E2', relative_weight=2.)
    assert m1 == m2
    assert float((l2-l1).detach()) == pytest.approx(2*m2['relative_huber'])
    l2.backward(); assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_native_ties_stay_and_invalid_never_selected():
    u = torch.tensor([[.9, .4], [.3, .4], [.2, .5]])
    valid = torch.tensor([[False, True], [True, True], [True, True]])
    torch.testing.assert_close(select_centers(u, valid, 1), torch.tensor([1, 2]))
    u[2, 1] = .4
    assert select_centers(u, valid, 1)[1] == 1


def test_error_is_deterministic_not_mutating_labels_or_depth():
    d = torch.full((1, 1, 28, 28), .5)
    before = d.clone()
    a = perturb_depth(d, 'smooth:10', 5); b = perturb_depth(d, 'smooth:10', 5)
    torch.testing.assert_close(a, b); torch.testing.assert_close(d, before)
    assert float((a-d).square().mean().sqrt()) == pytest.approx(.01, rel=1e-4)


class DepthMock(nn.Module):
    def __init__(self):
        super().__init__(); self.conv = nn.Conv2d(3, 8, 1)
    def forward(self, img, **kw):
        return (torch.full_like(img[:, :1], .5), None, None, None, self.conv(img), {})


class AdapterMock(nn.Module):
    def __init__(self):
        super().__init__(); self.conv = nn.Conv2d(8, 8, 1); self.out = nn.Conv2d(8, 3, 1)
    def forward(self, feat, ph, pw):
        x = self.conv(feat); return x, self.out(x)


class EnhancerMock(nn.Module):
    def __init__(self):
        super().__init__(); self.conv = nn.Conv2d(8, 8, 1)
    def forward(self, x, **kw):
        return self.conv(x) + kw['depth_map']*.01, {}


class GroupMock(nn.Module):
    def __init__(self):
        super().__init__(); self.conv = nn.Conv1d(8, 8, 1); self.config = SimpleNamespace()
    def forward(self, seed_features, seed_xyz, **kw):
        self.last_xyz = seed_xyz.detach()
        return self.conv(seed_features) + seed_xyz[..., 2][:, None]


class DecoderMock(nn.Module):
    def __init__(self):
        super().__init__(); self.conv = nn.Conv1d(8, 12, 1); self.width_head = nn.Conv1d(8, 2, 1)
    def forward(self, x, ep):
        q, a = ep['kview_angle_query_base_q'], ep['kview_angle_query_num_angle']
        raw = self.conv(x).reshape(1, 2, 6, q, a)
        mono = torch.cat([raw[:, :, :1], raw[:, :, :1] + torch.nn.functional.softplus(raw[:, :, 1:]).cumsum(2)], 2)
        return {'grasp_cdf_pred_angle_depth': mono.permute(0, 2, 3, 4, 1)}


def mock_reference():
    ref = nn.Module(); ref.depth_net = DepthMock(); ref.proposal_head = AdapterMock()
    ref.spatial_enhancer = EnhancerMock(); ref.kview_grasp_module = nn.Module()
    ref.kview_grasp_module.group = GroupMock(); ref.kview_grasp_module.decoder = DecoderMock()
    ref.num_angle = 3; ref.num_depth = 2; ref.camera_pose_key = 'camera_pose_vec'
    ref.kview_config = SimpleNamespace(head_model_dim=8)
    return ref


def mock_rotations(approach, angle):
    c, s = angle.cos(), angle.sin()
    out = torch.eye(3, device=angle.device).repeat(len(angle), 1, 1)
    out[:,1,1] = c; out[:,1,2] = -s; out[:,2,1] = s; out[:,2,2] = c
    return out


def test_online_joint_gradient_and_frozen_action_source(monkeypatch):
    stub = types.ModuleType('utils.label_generation'); stub.batch_viewpoint_params_to_matrix = mock_rotations
    monkeypatch.setitem(sys.modules, 'utils.label_generation', stub)
    torch.manual_seed(8)
    ref = mock_reference()
    model = CenterHypothesisCVA(ref, [-20,0,20]).train()
    assert not ref.training
    before = copy.deepcopy(ref.state_dict())
    batch = {'img': torch.randn(1,3,28,28), 'K': torch.eye(3)[None], 'camera_pose_vec': torch.ones(1,3)}
    act, valid = expand_centers(native_actions(), [-20,0,20])
    bundle = {'actions': act, 'valid': valid, 'token_ids': torch.tensor([0,7,30,140]),
              'view_xyz': torch.tensor([[1.,0,0]]).expand(4,3),
              'angle_ids': torch.tensor([0,1,2,0]), 'depth_ids': torch.tensor([0,1,0,1])}
    output, _, depth = model(batch, bundle, query_chunk=4)
    assert output.shape == (3,4,6)
    assert bool((output.diff(dim=-1) >= 0).all())
    target = torch.zeros_like(output)
    loss, _ = relative_cdf_loss(output, target, valid, 1, 'E2')
    loss.backward()
    for name in ('image_adapter','enhancer','group','decoder'):
        assert sum(float(p.grad.abs().sum()) for p in getattr(model,name).parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in ref.parameters())
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01); opt.step()
    for k, v in ref.state_dict().items():
        torch.testing.assert_close(v, before[k])
    restored = CenterHypothesisCVA(mock_reference(), [-20,0,20])
    restored.load_learned_state(model.learned_state())
    assert all(not k.startswith('reference.') for k in model.learned_state())


def test_depth_override_restored_on_failure():
    m = DepthMock(); original = m.forward
    with pytest.raises(RuntimeError):
        with _depth_override(m, ('replacement',)):
            assert m(None) == ('replacement',)
            raise RuntimeError('test')
    assert m.forward == original


def test_fingerprints_detect_changed_physical_width():
    a = native_actions().numpy(); before = array_sha(a[:,1:16]); a[0,1] += .001
    assert before != array_sha(a[:,1:16])
