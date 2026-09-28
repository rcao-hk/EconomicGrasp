"""CPU numerical contracts; no main CUDA extension imports."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metric_grasp_field_core import (
    MetricFieldConfig, MetricRayEvidenceHead, TaskFeatureAdapter, GraspFieldReadout,
    cdf_at, evidence_at, geometry_loss, gripper_support, project_points, sample_map,
)
from metric_field_runtime import (
    sampled_indices, move_batch, atomic_json, digest,
    validate_cdf_cpu_label_contract,
)


def wrapper_module():
    # main's models/__init__.py eagerly imports CUDA libraries. Load just this
    # new file for helper tests; full-main runtime is tested on the server.
    spec = importlib.util.spec_from_file_location("models.mgf_wrapper_for_test", ROOT/"models/economicgrasp_metric_field.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def actions(n=3, batch=1):
    a = torch.zeros(batch, n, 17)
    a[..., 0] = 1
    a[..., 1:4] = torch.tensor([.06, .02, .03])
    a[..., 4:13] = torch.eye(3).reshape(9)
    a[..., 15] = .5
    a[..., 16] = -1
    return a


def camera(batch=1):
    return torch.tensor([[80., 0., 31.5], [0., 80., 31.5], [0., 0., 1.]])[None].expand(batch, -1, -1)


@pytest.mark.parametrize("kwargs", [{"bins": 1}, {"hidden": 7}, {"evidence_mode": "bad"}, {"max_depth": .1}, {"surface_epsilon": 0}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        MetricFieldConfig(**kwargs)


def test_sampling_keeps_all_scenes_and_every_tenth_frame():
    idx = sampled_indices(25600, .1)
    assert len(idx) == 2600
    assert idx[:26] == list(range(0, 256, 10))
    assert len({i//256 for i in idx}) == 100
    assert len(sampled_indices(7680, .1)) == 780
    assert len(sampled_indices(7680, .1, 4)) == 4
    with pytest.raises(ValueError):
        sampled_indices(256, .13)


def test_continuous_depth_cdf_boundaries_and_interpolation():
    p = torch.tensor([[.2, .3, .5]]).expand(6, -1)
    z = torch.tensor([-1., 0., .5, 1., 2., 3.])
    got = cdf_at(p, z, 0., 3.)
    torch.testing.assert_close(got, torch.tensor([0., 0., .1, .2, .5, 1.]))


@pytest.mark.parametrize("mode", ["hard", "fixed", "learned"])
def test_surface_evidence_normalized_and_depth_detached(mode):
    cfg = MetricFieldConfig(bins=8, hidden=8, evidence_mode=mode)
    p = torch.softmax(torch.randn(4, 8), -1).requires_grad_()
    z = torch.tensor([.3, .4, .6, .8], requires_grad=True)
    e = evidence_at(p, z, cfg)
    torch.testing.assert_close(e[:, 1:].sum(-1), torch.ones(4))
    assert bool((e[:, 1:] >= -1e-6).all())
    # z is detached by the public readout boundary, not the pure formula.
    gp = torch.autograd.grad(e.sum(), p, allow_unused=True)[0]
    assert gp is None


def test_physical_transform_and_projection():
    a = actions(1)
    xyz, local, roles = gripper_support(a)
    assert xyz.shape == (1, 1, 27, 3)
    assert [int((roles == r).sum()) for r in range(5)] == [12, 4, 4, 3, 4]
    torch.testing.assert_close(xyz, local + a[..., 13:16].unsqueeze(-2))
    uv, valid = project_points(xyz, camera(), (64, 64))
    assert valid.all()
    assert torch.isfinite(uv).all()
    origin = torch.tensor([[[0., 0., .5]]])
    centre, _ = project_points(origin, camera(), (64, 64))
    torch.testing.assert_close(centre, torch.tensor([[[31.5, 31.5]]]))


def test_bilinear_map_pixel_convention():
    x = torch.arange(16).float().reshape(1, 1, 4, 4)
    uv = torch.tensor([[[0., 0.], [3., 3.], [1.5, 1.5]]])
    torch.testing.assert_close(sample_map(x, uv, (4, 4))[0, :, 0], torch.tensor([0., 15., 7.5]))


def test_geometry_trainable_grasp_geometry_detached_without_latent_bypass():
    torch.manual_seed(10)
    cfg = MetricFieldConfig(bins=8, hidden=8, action_chunk=2)
    metric = torch.nn.Conv2d(3, 5, 1)
    metric_depth = torch.nn.Conv2d(5, 1, 1)
    proposal = torch.nn.Conv2d(3, 7, 1)
    rgb = torch.randn(1, 3, 16, 16)
    mf = metric(rgb)
    depth = .2 + .6*metric_depth(mf).sigmoid()
    rf, ri = torch.randn(1, 6, 8, 8), torch.rand(1, 1, 16, 16)
    ray = MetricRayEvidenceHead(6, 5, cfg)
    adapter = TaskFeatureAdapter(7, 6, 5, 8)
    reader = GraspFieldReadout(cfg).train()
    logits, prob = ray(rf, ri, mf, depth, (64, 64), camera())
    task_feature = adapter(proposal(rgb), rf, mf, logits.shape[-2:])
    a = actions(3).requires_grad_()
    out = reader(task_feature, prob, a, camera(), (64, 64))
    assert out.shape == (1, 3, 6)
    assert bool((out[..., 1:] >= out[..., :-1]).all())
    task = out.square().mean()
    gp = torch.autograd.grad(task, [depth, logits, mf, a], retain_graph=True, allow_unused=True)
    assert all(g is None for g in gp)
    parameters = list(metric.parameters()) + list(metric_depth.parameters()) + list(ray.parameters())
    gs = torch.autograd.grad(task, parameters, retain_graph=True, allow_unused=True)
    assert all(g is None for g in gs)
    tg = torch.autograd.grad(task, list(adapter.parameters()) + list(reader.parameters()), retain_graph=True, allow_unused=True)
    assert sum(float(g.abs().sum()) for g in tg if g is not None) > 0
    geom = geometry_loss(logits, depth, torch.full_like(depth, .47), cfg)
    loss = sum(geom.values())
    gg = torch.autograd.grad(loss, parameters, allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in gg)
    assert sum(float(g.abs().sum()) for g in gg if g is not None) > 0


def test_empty_gt_loss_is_finite_zero_and_connected():
    cfg = MetricFieldConfig(bins=8, hidden=8)
    logits = torch.randn(1, 8, 4, 4, requires_grad=True)
    depth = torch.ones(1, 1, 8, 8, requires_grad=True)*.5
    gt = torch.full_like(depth, float("nan"))
    losses = geometry_loss(logits, depth, gt, cfg)
    assert all(torch.isfinite(v) and v == 0 for v in losses.values())
    sum(losses.values()).backward()


def test_cdf_discriminability_detects_perfect_and_reversed_ranking():
    m = wrapper_module()
    # Four valid candidates: two negatives (bin=0), two positives (bin=1).
    bins = torch.tensor([[[[0, 1, 0, 1]]]], dtype=torch.long)
    valid = torch.ones_like(bins, dtype=torch.bool)
    # [B,T,Q,A,D], T=2.
    perfect = torch.tensor(
        [[[[[-8.0, 8.0, -7.0, 7.0]]],
          [[[-8.0, 8.0, -7.0, 7.0]]]]]
    )
    reversed_logits = -perfect

    good = m._cdf_discriminability_stats(
        perfect, bins, valid, prefix="field", histogram_bins=64
    )
    bad = m._cdf_discriminability_stats(
        reversed_logits, bins, valid, prefix="field", histogram_bins=64
    )

    assert float(good["field_cdf_pred_pos_mean"]) > float(
        good["field_cdf_pred_neg_mean"]
    )
    assert float(good["field_cdf_pos_neg_gap"]) > 0.9
    assert float(good["field_cdf_any_success_auroc64"]) == pytest.approx(1.0)
    assert float(good["field_cdf_any_success_auprc64"]) == pytest.approx(1.0)
    assert float(good["field_cdf_any_success_auprc_lift64"]) == pytest.approx(2.0)
    assert float(good["field_cdf_utility_pearson"]) > 0.99

    assert float(bad["field_cdf_pos_neg_gap"]) < -0.9
    assert float(bad["field_cdf_any_success_auroc64"]) == pytest.approx(0.0)
    assert float(bad["field_cdf_any_success_auprc64"]) == pytest.approx(0.5)
    assert float(bad["field_cdf_utility_pearson"]) < -0.99


def test_cdf_discriminability_handles_one_class_without_nan():
    m = wrapper_module()
    bins = torch.zeros(1, 1, 1, 4, dtype=torch.long)
    valid = torch.ones_like(bins, dtype=torch.bool)
    logits = torch.zeros(1, 2, 1, 1, 4)
    stats = m._cdf_discriminability_stats(
        logits, bins, valid, prefix="field", histogram_bins=64
    )
    assert all(torch.isfinite(v) for v in stats.values())
    assert float(stats["field_cdf_positive_fraction"]) == 0.0
    assert float(stats["field_cdf_any_success_auroc64"]) == 0.0
    assert float(stats["field_cdf_any_success_auprc64"]) == 0.0


def test_query_listwise_ranking_prefers_correct_within_query_order():
    m = wrapper_module()
    # One query, four angle-depth candidates. Target utilities from cdf bins:
    # [0, 0.5, 1.0, 0] for T=2.
    bins = torch.tensor([[[[0, 2, 1, 0]]]], dtype=torch.long)
    valid = torch.ones_like(bins, dtype=torch.bool)

    # Correct ordering: candidate 2 > candidate 1 > candidates 0/3.
    good = torch.tensor(
        [[[[[-8.0, 0.0, 8.0, -8.0]]],
          [[[-8.0, 0.0, 8.0, -8.0]]]]],
        requires_grad=True,
    )
    # Reversed ordering.
    bad = torch.tensor(
        [[[[[8.0, 0.0, -8.0, 8.0]]],
          [[[8.0, 0.0, -8.0, 8.0]]]]],
        requires_grad=True,
    )

    good_loss, good_stats = m.compute_query_listwise_ranking_loss(
        good, bins, valid, temperature=0.1
    )
    bad_loss, bad_stats = m.compute_query_listwise_ranking_loss(
        bad, bins, valid, temperature=0.1
    )

    assert torch.isfinite(good_loss)
    assert torch.isfinite(bad_loss)
    assert float(good_loss.detach()) < float(bad_loss.detach())
    assert float(good_stats["ranking_informative_query_fraction"]) == 1.0
    assert float(good_stats["ranking_top1_best_hit"]) == 1.0
    assert float(good_stats["ranking_selection_regret"]) == 0.0
    assert float(bad_stats["ranking_top1_best_hit"]) == 0.0
    assert float(bad_stats["ranking_selection_regret"]) > 0.0

    good_loss.backward()
    assert good.grad is not None
    assert torch.isfinite(good.grad).all()
    assert float(good.grad.abs().sum()) > 0.0


def test_query_listwise_ranking_skips_constant_target_queries():
    m = wrapper_module()
    bins = torch.zeros(1, 1, 1, 4, dtype=torch.long)
    valid = torch.ones_like(bins, dtype=torch.bool)
    logits = torch.zeros(1, 2, 1, 1, 4, requires_grad=True)

    loss, stats = m.compute_query_listwise_ranking_loss(
        logits, bins, valid, temperature=0.1
    )
    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0
    assert float(stats["ranking_informative_query_fraction"]) == 0.0
    assert float(stats["ranking_informative_query_count"]) == 0.0
    loss.backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


def test_query_listwise_ranking_respects_invalid_candidates():
    m = wrapper_module()
    bins = torch.tensor([[[[0, 1, 0, 0]]]], dtype=torch.long)
    valid = torch.tensor([[[[True, True, False, False]]]])
    logits = torch.tensor(
        [[[[[-4.0, 4.0, 20.0, 20.0]]],
          [[[-4.0, 4.0, 20.0, 20.0]]]]],
        requires_grad=True,
    )
    loss, stats = m.compute_query_listwise_ranking_loss(
        logits, bins, valid, temperature=0.1
    )
    assert torch.isfinite(loss)
    assert float(stats["ranking_top1_best_hit"]) == 1.0
    assert float(stats["ranking_selection_regret"]) == 0.0


def test_main_metric_loss_normalization_preserved():
    cfg = MetricFieldConfig(bins=8, hidden=8)
    logits = torch.zeros(1, 8, 2, 2, requires_grad=True)
    depth = torch.full((1, 1, 2, 2), .5, requires_grad=True)
    gt = torch.tensor([[[[.6, 0.], [0., 0.]]]])
    result = geometry_loss(logits, depth, gt, cfg)
    assert float(result["depth_l1"].detach()) == pytest.approx(.025)


def test_chunking_and_query_permutation_preserve_field():
    torch.manual_seed(1)
    cfg = MetricFieldConfig(bins=8, hidden=8, action_chunk=1)
    model = GraspFieldReadout(cfg).eval()
    feat, prob = torch.randn(1, 8, 16, 16), torch.randn(1, 8, 16, 16).softmax(1)
    a = actions(4)
    a[0, :, 15] = torch.tensor([.45, .5, .55, .6])
    k = camera()
    one = model(feat, prob, a, k, (64, 64))
    model.cfg = replace(cfg, action_chunk=4)
    all_at_once = model(feat, prob, a, k, (64, 64))
    torch.testing.assert_close(one, all_at_once, atol=1e-6, rtol=1e-5)
    order = torch.tensor([2, 0, 3, 1])
    shuffled = model(feat, prob, a[:, order], k, (64, 64))
    torch.testing.assert_close(shuffled, all_at_once[:, order])


def test_out_of_view_support_has_no_nan():
    cfg = MetricFieldConfig(bins=8, hidden=8, action_chunk=4)
    model = GraspFieldReadout(cfg).eval()
    a = actions(2)
    a[..., 13] = 1000.
    out = model(torch.randn(1, 8, 16, 16), torch.ones(1, 8, 16, 16)/8,
                a, camera(), (64, 64))
    assert torch.isfinite(out).all()


def test_empty_object_payloads_are_filtered_consistently():
    m = wrapper_module()
    # Object slot 1 is present in frame metadata but has no economic-grasp rows.
    batch = {
        "object_poses_list": [[torch.eye(4)[:3], torch.eye(4)[:3]]],
        "grasp_points_list": [[torch.ones(2, 3), torch.empty(0, 3)]],
        "grasp_rotations_list": [[torch.ones(2, 4), torch.empty(0, 4)]],
        "grasp_depth_list": [[torch.ones(2), torch.empty(0)]],
        "grasp_widths_list": [[torch.ones(2), torch.empty(0)]],
        "grasp_scores_list": [[torch.ones(2), torch.empty(0)]],
        "view_graspness_list": [[torch.ones(2, 300), torch.empty(0, 300)]],
        "top_view_index_list": [[torch.ones(2, 5, dtype=torch.long), torch.empty(0, 5, dtype=torch.long)]],
        "grasp_collision_list": [[torch.ones(2), torch.empty(0)]],
        "grasp_cdf_bins_list": [[torch.ones(2, 5, 12, 4, dtype=torch.uint8), torch.empty(0, 5, 12, 4, dtype=torch.uint8)]],
        "grasp_widths_depth_list": [[torch.ones(2, 5, 12, 4), torch.empty(0, 5, 12, 4)]],
        "grasp_width_valids_depth_list": [[torch.ones(2, 5, 12, 4, dtype=torch.bool), torch.empty(0, 5, 12, 4, dtype=torch.bool)]],
    }
    out, report = m.filter_empty_grasp_objects(batch)
    assert report == [{
        "batch_index": 0,
        "objects_before": 2,
        "objects_after": 1,
        "dropped_object_slots": [1],
    }]
    for key in m._OBJECT_PAYLOAD_KEYS:
        assert len(out[key][0]) == 1
    # Caller-owned batch must not be mutated.
    assert len(batch["object_poses_list"][0]) == 2
    assert len(batch["grasp_points_list"][0]) == 2


def test_object_payload_cpu_contract_rejects_non_cpu_tensor():
    m = wrapper_module()
    batch = {
        "object_poses_list": [[torch.eye(4)[:3]]],
        "grasp_points_list": [[torch.empty(1, 3, device="meta")]],
    }
    with pytest.raises(RuntimeError, match="must remain CPU-resident"):
        m.assert_object_payloads_cpu(batch)


def test_all_empty_object_payloads_still_fail():
    m = wrapper_module()
    batch = {
        "object_poses_list": [[torch.eye(4)[:3]]],
        "grasp_points_list": [[torch.empty(0, 3)]],
    }
    with pytest.raises(RuntimeError, match="all economic-grasp object caches are empty"):
        m.filter_empty_grasp_objects(batch)


def test_candidate_actions_match_main_decode_width_and_insertion():
    m = wrapper_module()
    ep = {"xyz_graspable": torch.tensor([[[0., 0., .6]]], requires_grad=True),
          "grasp_top_view_xyz": torch.tensor([[[0., 0., -1.]]]),
          "grasp_width_pred_angle_depth": torch.full((1, 4, 1, 2), .5, requires_grad=True)}
    def rotation(v, angles):
        # Here only test that precisely the main decoder's arguments are used.
        torch.testing.assert_close(v, torch.tensor([[0., 0., 1.]]).expand(2, -1))
        torch.testing.assert_close(angles, torch.tensor([0., torch.pi/2]))
        return torch.eye(3)[None].expand(len(v), -1, -1)
    a, shape = m.build_candidate_actions(ep, rotation)
    assert shape == (1, 1, 2, 4)
    assert not a.requires_grad
    torch.testing.assert_close(a[0, :, 1], torch.full((8,), .06))
    torch.testing.assert_close(a[0, :, 3], torch.tensor([.01, .02, .03, .04]*2))
    torch.testing.assert_close(a[0, :, 15], torch.full((8,), .6))


def _minimal_cdf_batch(width_dtype=torch.uint16):
    return {
        "object_poses_list": [[torch.eye(4)[:3]]],
        "grasp_points_list": [[torch.ones(2, 3)]],
        "view_graspness_list": [[torch.ones(2, 300)]],
        "top_view_index_list": [[torch.ones(2, 5, dtype=torch.int32)]],
        "grasp_cdf_bins_list": [[torch.ones(2, 5, 12, 4, dtype=torch.uint8)]],
        "grasp_widths_depth_list": [[torch.ones(2, 5, 12, 4, dtype=width_dtype)]],
        "grasp_width_valids_depth_list": [[torch.ones(2, 5, 12, 4, dtype=torch.uint8)]],
        "cdf_thresholds": torch.tensor([[.2, .4, .6, .8, 1., 1.2]]),
    }


def test_cdf_width_cache_must_stay_uint16_millimetres():
    good = _minimal_cdf_batch(torch.uint16)
    validate_cdf_cpu_label_contract(good)
    bad = _minimal_cdf_batch(torch.float32)
    # This reproduces the old direct-dataset path: widths had already been
    # divided by 1000 before the matcher applied its own mm->m conversion.
    with pytest.raises(TypeError, match="raw uint16 millimetres"):
        validate_cdf_cpu_label_contract(bad)


def test_batch_annotation_payload_stays_host_and_infer_drops_depth():
    batch = _minimal_cdf_batch()
    payload = batch["object_poses_list"]
    batch.update({
        "img": torch.ones(1, 3, 4, 4),
        "K": camera(),
        "gt_depth_m": torch.ones(1, 1, 4, 4),
        "sensor_depth_m": torch.ones(1, 1, 4, 4),
    })
    train = move_batch(batch, "cpu")
    assert train["object_poses_list"] is payload
    assert train["grasp_widths_depth_list"][0][0].device.type == "cpu"
    infer = move_batch(batch, "cpu", inference=True)
    assert set(infer) == {"img", "K"}


@pytest.mark.parametrize("script", ["train_metric_grasp_field.py", "inference_metric_grasp_field.py", "eval_metric_grasp_field.py"])
def test_cli_help_does_not_import_main_cuda_dependencies(script):
    p = subprocess.run([sys.executable, str(ROOT/script), "--help"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "usage:" in p.stdout


def test_mixed_matched20_sampling_counts():
    # 100 train scenes x 256 frames, stride 5 -> 52 frames/scene.
    assert len(sampled_indices(100 * 256, .2)) == 5200
    # 30 Seen scenes x 256 frames, stride 10 -> 26 frames/scene.
    assert len(sampled_indices(30 * 256, .1)) == 780


def test_training_cli_has_no_gradient_accumulation_option():
    p = subprocess.run(
        [sys.executable, str(ROOT/"train_metric_grasp_field.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    assert "--batch-size" in p.stdout
    assert "--train-fraction" in p.stdout
    assert "--eval-fraction" in p.stdout
    assert "--use-fuse-depth" in p.stdout
    assert "--lr-schedule" in p.stdout
    assert "--grad-clip" in p.stdout
    assert "--grad-accum" not in p.stdout


def test_inference_cli_exposes_optional_collision_filter():
    p = subprocess.run(
        [sys.executable, str(ROOT/"inference_metric_grasp_field.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    assert "--score-source" in p.stdout
    assert "--collision-thresh" in p.stdout
    assert "--collision-voxel-size" in p.stdout
    assert "--collision-approach-dist" in p.stdout


def test_atomic_json_and_digest(tmp_path):
    payload = {"version": "test", "sampling": sampled_indices(256, .1)}
    atomic_json(tmp_path/"protocol.json", payload)
    import json
    assert digest(json.loads((tmp_path/"protocol.json").read_text())) == digest(payload)

@pytest.fixture
def fake_main(monkeypatch, tmp_path):
    """Exercise the NEW wrapper/loss while replacing unavailable main CUDA stack.

    This is a mock integration test, not evidence that GraspNet GPU smoke passed.
    """
    import types
    from torch import nn
    monkeypatch.chdir(tmp_path)
    (tmp_path/"checkpoints").mkdir()
    m = wrapper_module()
    pkg = types.ModuleType("models")
    pkg.__path__ = [str(ROOT/"models")]
    monkeypatch.setitem(sys.modules, "models", pkg)
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    from metric_field_runtime import BASE_CONFIG
    cfg = types.SimpleNamespace(**BASE_CONFIG)
    cfg.num_angle = 2
    args_mod = types.ModuleType("utils.arguments")
    args_mod.cfgs = cfg
    monkeypatch.setitem(sys.modules, "utils.arguments", args_mod)

    class FakeDPT(nn.Module):
        def __init__(self, in_channels, features=128, out_dim=1, **kwargs):
            super().__init__()
            self.feature = nn.Conv2d(1, features, 1)
            self.output = nn.Conv2d(features, out_dim, 1)
        def forward(self, feats, patch_h, patch_w):
            feature = self.feature(feats[0])
            return feature, self.output(feature)
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = nn.Parameter(torch.tensor(.4), requires_grad=False)
            self.calls = 0
        def forward(self, img):
            self.calls += 1
            return [img.mean(1, keepdim=True).detach() * self.gain]
    class Depth(nn.Module):
        def __init__(self):
            super().__init__()
            self.depthnet = nn.Module()
            self.depthnet.pretrained = Encoder()
            self.depthnet.depth_head = FakeDPT(768, features=128)
        def forward(self, img):
            f = self.depthnet.pretrained(img)
            mf, raw = self.depthnet.depth_head(f, 2, 2)
            d = raw.sigmoid()
            return d, d, mf, raw, f, {}
    class Base(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.depth_net = Depth()
            self.proposal_head = FakeDPT(768, features=128, out_dim=3)
            self.width = nn.Parameter(torch.tensor(.5))
            self.cdf = nn.Parameter(torch.zeros(1, 6, 2, 2, 4))
            self.view = nn.Linear(1, 1)
            self.view.is_training = True
            self.is_training = True
        def forward(self, ep):
            depth, _, mf, raw, feats, _ = self.depth_net(ep["img"])
            pf, pred = self.proposal_head(feats, 2, 2)
            b = len(depth)
            zero = depth.mean((1, 2, 3))*0
            z = depth.mean((1, 2, 3))
            ep["xyz_graspable"] = torch.stack([zero, zero, z], -1)[:, None].expand(-1, 2, -1)
            ep["grasp_top_view_xyz"] = depth.new_tensor([0., 0., -1.])[None, None].expand(b, 2, 3)
            ep["grasp_width_pred_angle_depth"] = self.width.expand(b, 4, 2, 2)
            # Deliberately put numeric depth + metric latent in a main grasp path;
            # the wrapper must eliminate even these graph bypasses.
            ep["grasp_cdf_pred_angle_depth"] = self.cdf.expand(b, -1, -1, -1, -1) + (pf.mean()+depth.mean()+mf.mean())*.001
            ep["objectness_score"] = pred[:, :2]
            ep["graspness_score"] = pred[:, 2:]
            ep["view_score"] = self.view(pf.mean().reshape(1, 1))
            ep["batch_grasp_cdf_valid_mask"] = torch.ones(b, 2, 2, 4, dtype=torch.bool)
            return ep
    dpt_mod = types.ModuleType("models.dinov2_dpt")
    dpt_mod.DPTHead = FakeDPT
    monkeypatch.setitem(sys.modules, "models.dinov2_dpt", dpt_mod)
    base_mod = types.ModuleType("models.economicgrasp_bip3d")
    base_mod.economicgrasp_dpt = Base
    monkeypatch.setitem(sys.modules, "models.economicgrasp_bip3d", base_mod)
    labels = types.ModuleType("utils.label_generation")
    labels.batch_viewpoint_params_to_matrix = lambda v, a: torch.eye(3, device=v.device)[None].expand(len(v), -1, -1)
    monkeypatch.setitem(sys.modules, "utils.label_generation", labels)
    losses = types.ModuleType("models.loss_economicgrasp_depth_kview_transformer")
    def mse(key):
        return lambda ep: (ep[key].square().mean(), ep)
    losses.compute_objectness_loss_tok = mse("objectness_score")
    losses.compute_graspness_loss_tok = mse("graspness_score")
    losses.compute_view_graspness_loss = mse("view_score")
    losses.compute_cva_width_depth_loss = mse("grasp_width_pred_angle_depth")
    losses.compute_cva_cdf_loss = lambda ep, balanced=False: (
        torch.nn.functional.binary_cross_entropy_with_logits(ep["grasp_cdf_pred_angle_depth"],
                                                            torch.ones_like(ep["grasp_cdf_pred_angle_depth"])), ep)
    monkeypatch.setitem(sys.modules, "models.loss_economicgrasp_depth_kview_transformer", losses)
    original = FakeDPT(768, features=128)
    official = {"pretrained.gain": torch.tensor(.4), **{"depth_head."+k: v for k, v in original.state_dict().items()}}
    torch.save(official, tmp_path/"checkpoints/depth_anything_v2_vitb.pth")
    return m, original, tmp_path


def test_mock_main_wrapper_reuses_encoder_restores_decoder_and_detaches(fake_main):
    m, original, tmp_path = fake_main
    from train_metric_grasp_field import gradient_contract
    cfg = MetricFieldConfig(bins=8, hidden=8, field_stride=4, action_chunk=8)
    model = m.EconomicGraspMetricField(cfg)
    model.train()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(model.relative_decoder.state_dict()[key], value)
    assert not model.relative_decoder.training
    assert all(not p.requires_grad for p in model.relative_decoder.parameters())
    batch = {"img": torch.randn(1, 3, 64, 64), "K": camera(),
             "gt_depth_m": torch.full((1, 1, 64, 64), .52), "object_poses_list": [[]]}
    report = gradient_contract(model, batch, m.metric_field_loss,
                               {"profile_weight": 1., "profile_mean_weight": 10., "base_cdf_weight": .25})
    assert report["grasp_to_geometry_parameters_norm"] == 0
    assert report["grasp_to_predicted_depth_norm"] == 0
    assert report["grasp_to_depth_profile_norm"] == 0
    assert report["geometry_to_geometry_parameters_norm"] > 0
    assert report["grasp_to_field_reader_norm"] > 0
    assert model.base.depth_net.depthnet.pretrained.calls == 1
    assert model._depth_pack is None and model._proposal_feature is None
    model.eval()
    assert not model.base.is_training and not model.base.view.is_training
    with torch.no_grad():
        ep = model({"img": batch["img"], "K": batch["K"]})
    assert ep["grasp_cdf_pred_angle_depth"].shape == (1, 6, 2, 2, 4)
    assert ep["cva_force_process_grasp_labels"] is False


def test_mock_online_optimizer_updates_metric_head_but_not_frozen_prior(fake_main):
    m, _, tmp_path = fake_main
    model = m.EconomicGraspMetricField(MetricFieldConfig(bins=8, hidden=8, action_chunk=8)).train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    batch = {"img": torch.randn(1, 3, 64, 64), "K": camera(),
             "gt_depth_m": torch.full((1, 1, 64, 64), .52), "object_poses_list": [[]]}
    loss, _ = m.metric_field_loss(model(batch), model.config)
    loss.backward()
    opt.step()
    changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
    assert any(k.startswith("base.depth_net.depthnet.depth_head") for k in changed)
    assert any(k.startswith("ray_head") for k in changed)
    assert any(k.startswith("readout") for k in changed)
    assert not any(k.startswith("relative_decoder") or ".pretrained." in k for k in changed)
    # Full-wrapper checkpoint round trip including the restored frozen decoder.
    path = tmp_path/"new.pt"
    torch.save({"model": model.state_dict()}, path)
    clone = m.EconomicGraspMetricField(model.config)
    clone.load_state_dict(torch.load(path, weights_only=True)["model"], strict=True)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(clone.state_dict()[k], v)


def test_shared_manifest_allows_identical_inference_shards(tmp_path):
    from metric_field_runtime import ensure_manifest
    path = tmp_path/"test_novel/protocol.json"
    ensure_manifest(path, {"checkpoint": "abc", "schedule": [0, 10]})
    ensure_manifest(path, {"checkpoint": "abc", "schedule": [0, 10]})
    with pytest.raises(RuntimeError):
        ensure_manifest(path, {"checkpoint": "different", "schedule": [0, 10]})


def test_batch_two_camera_query_dimensions():
    cfg = MetricFieldConfig(bins=8, hidden=8, action_chunk=2)
    head = MetricRayEvidenceHead(6, 5, cfg)
    logits, prob = head(torch.randn(2, 6, 8, 8), torch.rand(2, 1, 16, 16),
                        torch.randn(2, 5, 16, 16), torch.full((2, 1, 16, 16), .5),
                        (64, 64), camera(2))
    readout = GraspFieldReadout(cfg).eval()
    result = readout(torch.randn(2, 8, 16, 16), prob, actions(3, 2), camera(2), (64, 64))
    assert result.shape == (2, 3, 6)
    assert torch.isfinite(result).all()
