"""Small real attention/decoder tests; no dataset or pretrained model required."""
import unittest
import sys
from types import SimpleNamespace

import torch
from torch import nn

_argv = sys.argv
try:
    sys.argv = [sys.argv[0]]  # Legacy models package imports the training CLI.
    from models.grasp_designs import install_grasp_designs, validate_checkpoint_design
    from models.kview_query_transformer import CenterViewAngleCDFCandidateDecoder
finally:
    sys.argv = _argv


class OriginalGroup(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(8, 16, 1)

    def forward(self, seed_features, **kwargs):
        return self.proj(seed_features)


def fixture():
    model = nn.Module()
    model.use_obs_depth = model.use_gt_xyz_for_train = False
    model.geometry_depth_source = "pred"
    model.use_cdf = True
    model.kview_config = SimpleNamespace(mode="A1")
    model.seed_feature_dim = 8
    model.kview_grasp_module = nn.Module()
    model.kview_grasp_module.group = OriginalGroup()
    model.kview_grasp_module.decoder = CenterViewAngleCDFCandidateDecoder(
        2, 4, in_dim=16, hidden_dim=32, num_layers=1)
    return model


class MainGraspDesignTests(unittest.TestCase):
    def test_disabled_preserves_modules_rng_and_checkpoint(self):
        model = fixture()
        modules = dict(model.named_modules())
        state = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        install_grasp_designs(model)
        self.assertEqual(modules, dict(model.named_modules()))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(state.keys(), model.state_dict().keys())
        for k, v in state.items():
            torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)
        validate_checkpoint_design(model, {})  # Legacy checkpoints remain valid.

    def test_all_four_switch_combinations_forward_backward(self):
        for ray, gvar in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(ray=ray, gvar=gvar):
                torch.manual_seed(7)
                model = fixture()
                install_grasp_designs(model, enable_point_rayrope=ray, enable_gvar=gvar,
                                      rayrope_group_chunk=2, gvar_action_chunk=4)
                group = model.kview_grasp_module.group
                appearance = torch.randn(1, 8, 16, 16, requires_grad=True)
                depth = torch.full((1, 1, 64, 64), .5, requires_grad=True)
                centers = torch.tensor([[[0., 0., .5]]]).expand(1, 4, 3).clone().requires_grad_()
                ep = dict(kview_angle_query_base_q=2, kview_angle_query_num_angle=2)
                if gvar:
                    ep['_gvar_pregeom'] = appearance
                grouped = group(
                    seed_features=torch.randn(1, 8, 4, requires_grad=True),
                    token_sel_idx=torch.full((1, 4), 32 * 64 + 32, dtype=torch.long),
                    seed_xyz=centers,
                    top_view_rot=torch.eye(3).reshape(1, 1, 3, 3).expand(1, 4, 3, 3),
                    feat_map=appearance, depth_map=depth,
                    camera_K=torch.tensor([[[160., 0., 31.5], [0., 160., 31.5], [0., 0., 1.]]]),
                    end_points=ep)
                result = model.kview_grasp_module.decoder(grouped, ep)
                widths = result['grasp_width_pred_angle_depth']
                scores = result['grasp_cdf_pred_angle_depth']
                self.assertTrue(torch.isfinite(widths).all() and torch.isfinite(scores).all())
                (widths.square().mean() + scores.square().mean()).backward()
                self.assertIsNone(depth.grad)
                self.assertIsNone(centers.grad)
                if ray or gvar:
                    self.assertGreater(appearance.grad.abs().sum().item(), 0)
                self.assertNotIn('_gvar_context', ep)
                self.assertNotIn('_gvar_pregeom', ep)
                validate_checkpoint_design(model, {'grasp_design': model.grasp_design})
                if ray or gvar:
                    with self.assertRaises(ValueError):
                        validate_checkpoint_design(model, {})

    def test_reject_privileged_geometry_and_legacy_labels(self):
        for field, value in [('use_obs_depth', True), ('geometry_depth_source', 'gt'), ('use_cdf', False)]:
            model = fixture()
            setattr(model, field, value)
            with self.assertRaises(ValueError):
                install_grasp_designs(model, enable_point_rayrope=True)


if __name__ == '__main__':
    unittest.main()
