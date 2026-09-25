#!/usr/bin/env python3
"""Exercise discrete replay, analytic FD agreement and state restoration on CPU.

The small model deliberately has depth-dependent seed IDs, nearest labels and
attention masks. This tests the replay controller, not real-model connectivity.
"""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depth_dynamics import capture_rng_state
from depth_dynamics_directional import run_directional_probe, LOSS_KEYS


def knn_points(x, y, K=1):
    distance = torch.cdist(x, y)
    value, index = distance.topk(K, largest=False)
    return value, index, None


def process_grasp_labels_cdf_width(ep):
    xyz = ep["xyz_graspable"]
    B, Q, _ = xyz.shape
    points = xyz.new_tensor([[[0.0, 0.0, 0.45], [0.0, 0.0, 0.55]]])
    nearest = []
    ids = []
    for i in range(B):
        _, index, _ = knn_points(xyz[i].unsqueeze(0), points)
        index = index.reshape(-1)
        ids.append(index)
        nearest.append(points[0, index])
    ids = torch.stack(ids)
    valid = torch.ones(B, Q, 1, 1, dtype=torch.bool)
    ep.update({
        "batch_grasp_point": torch.stack(nearest),
        "batch_grasp_view_graspness": ids.to(xyz.dtype).unsqueeze(-1).expand(B, Q, 2),
        "batch_valid_mask": valid[..., 0, 0],
        "batch_grasp_cdf_bins_angle_depth": ids[..., None, None],
        "batch_grasp_cdf_valid_mask": valid,
        "batch_grasp_cdf_pos_mask": valid & (ids[..., None, None] > 0),
        "batch_grasp_width_angle_depth": ids[..., None, None].to(xyz.dtype) * 0.01,
        "batch_grasp_width_valid_mask_angle_depth": valid,
        "batch_grasp_cdf_thresholds": xyz.new_tensor([0.2]),
        "C: Valid Points": xyz.new_tensor(float(Q)),
    })
    return torch.eye(3, dtype=xyz.dtype).expand(B, Q, 3, 3), ep


class Depth(nn.Module):
    def forward(self, image):
        _ = torch.rand(())  # normal depth path consumes RNG
        return image, image, image, image, [image], {}


class View(nn.Module):
    def _select_top_view_inds(self, scores):
        _ = torch.rand(())  # forcing views must not remove this draw
        return scores.argmax(-1)


class Selector(nn.Module):
    def _select_view_indices(self, scores, is_training, forced_view_inds=None):
        _ = torch.rand(())
        probability = scores.softmax(-1)
        ids = scores.argmax(-1).unsqueeze(-1)
        if forced_view_inds is not None:
            ids = forced_view_inds.unsqueeze(-1)
        return ids, torch.zeros_like(ids), probability.gather(-1, ids), 1, probability


class Group(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _make_view_conditioned_grid(self, xyz):
        return xyz[..., 2], xyz[..., 2] >= 0.4999


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.seed_selection_mode = "image_fps"
        self.geometry_depth_source = "pred"
        self.use_obs_depth = self.use_gt_xyz_for_train = False
        self.use_cdf = self.is_training = True
        self.depth_net = Depth()
        self.view = View()
        self.spatial_enhancer = nn.Identity()
        self.kview_config = SimpleNamespace(detach_depth=True)
        self.kview_grasp_module = nn.Module()
        self.kview_grasp_module.config = self.kview_config
        self.kview_grasp_module.group = Group(self.kview_config)
        self.kview_grasp_module.selector = Selector()
        self.readout = nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
        self.register_buffer("counter", torch.zeros(()))
        self._vis_iter = 3
        self.set_depth_grad_routes("none")

    def set_depth_grad_routes(self, value):
        assert value in ("none", "all")
        self.depth_grad_routes = value
        self.spatial_enhancer.detach_depth_grad = value == "none"
        self.detach_seed_xyz_grad = value == "none"
        self.kview_config.detach_depth = value == "none"

    def get_depth_grad_routes(self):
        return dict(gse=not self.spatial_enhancer.detach_depth_grad,
                    seed_xyz=not self.detach_seed_xyz_grad,
                    support=not self.kview_config.detach_depth)

    def forward(self, ep):
        self.counter += 1
        self._vis_iter += 1
        z = self.depth_net(ep["img"])[0]
        # This discontinuous seed choice changes across a mu perturbation.
        first = 0 if bool(z[0, 0, 0, 0] >= 0.5) else 1
        ids = torch.tensor([[first, 3]])
        ids = ep.get("image_fps_seed_idx_override", ids)
        zs = z.flatten(1).gather(1, ids)
        xyz = torch.stack([zs * 0, zs * 0, zs], -1)
        noise = torch.rand((), dtype=z.dtype) * 0.05
        score = torch.stack([zs - 0.5 + noise, 0.5 - zs], -1)
        base_views = self.view._select_top_view_inds(score)
        ep.update(xyz_graspable=xyz, grasp_top_view_inds=base_views)
        _, ep = process_grasp_labels_cdf_width(ep)
        base_labels = ep["batch_grasp_view_graspness"]
        queries = self.kview_grasp_module.selector._select_view_indices(
            score, True, base_views)[0].squeeze(-1)
        ep["grasp_top_view_inds"] = queries
        _, ep = process_grasp_labels_cdf_width(ep)
        grid, valid = self.kview_grasp_module.group._make_view_conditioned_grid(xyz)
        continuous = (grid.square() + zs * 0.1) * valid.to(z.dtype)
        ep.update(depth_map_pred=z, kview_base_token_sel_idx=ids, token_sel_idx=ids,
                  xyz_graspable=xyz, grasp_top_view_inds=queries,
                  view_score=score, batch_grasp_view_graspness=base_labels,
                  toy_continuous=continuous, toy_readout=self.readout)
        return ep


def loss_fn(ep):
    raw = {
        "depth": (ep["depth_map_pred"] - ep["gt_depth_m"]).square().mean(),
        "objectness": ep["toy_readout"].square(),
        "graspness": ep["toy_readout"] * 0.2,
        "view": (ep["view_score"] - ep["batch_grasp_view_graspness"]).square().mean(),
        "cdf": (ep["toy_continuous"] - ep["batch_grasp_cdf_bins_angle_depth"][..., 0, 0]).square().mean(),
        "width": (ep["toy_continuous"] - ep["batch_grasp_width_angle_depth"][..., 0, 0]).square().mean(),
    }
    for name, key in LOSS_KEYS.items():
        ep[key] = raw[name]
    return sum(raw.values()), ep


def main():
    torch.manual_seed(813)
    model = Toy()
    model.readout.grad = torch.tensor(1.7, dtype=torch.float64)
    grad_identity = model.readout.grad
    before = capture_rng_state()
    label_identity = process_grasp_labels_cdf_width
    batch = {"img": torch.tensor([[[[0.50001, 0.49], [0.6, 0.49991]]]], dtype=torch.float64),
             "gt_depth_m": torch.full((1, 1, 2, 2), 0.52, dtype=torch.float64)}
    rows = run_directional_probe(model, batch, loss_fn,
                                 loss_weights={name: 1.0 for name in LOSS_KEYS})
    frozen = [r for r in rows if r["protocol"] == "frozen_continuous"]
    errors = [r["frozen_fd_abs_error"] for r in frozen if r["frozen_fd_abs_error"] is not None]
    small_errors = [r["frozen_fd_abs_error"] for r in frozen
                    if r["relative_step"] == 1e-4 and r["frozen_fd_abs_error"] is not None]
    assert max(small_errors) < 1e-6, max(small_errors)
    assert any(r["plus_switch_stats"]["seed_switch_rate_aligned_slots"] > 0
               or r["minus_switch_stats"]["seed_switch_rate_aligned_slots"] > 0
               for r in rows if r["protocol"] == "native_recompute")
    assert all(r["frozen_fd_abs_error"] is None for r in rows if r["protocol"] == "native_recompute")
    assert not any(model.get_depth_grad_routes().values())
    assert model.counter.item() == 0 and model._vis_iter == 3
    assert model.readout.grad is grad_identity and model.readout.grad.item() == 1.7
    assert process_grasp_labels_cdf_width is label_identity
    assert not model.depth_net._forward_hooks
    assert "_select_top_view_inds" not in vars(model.view)
    assert torch.equal(before["torch"], capture_rng_state()["torch"])

    def failed_loss(ep):
        raise RuntimeError("intentional replay failure")

    try:
        run_directional_probe(model, batch, failed_loss,
                              loss_weights={name: 1.0 for name in LOSS_KEYS})
    except RuntimeError as exc:
        assert str(exc) == "intentional replay failure"
    else:
        raise AssertionError("failure injection did not run")
    assert not any(model.get_depth_grad_routes().values())
    assert model.counter.item() == 0 and model._vis_iter == 3
    assert model.readout.grad is grad_identity and model.readout.grad.item() == 1.7
    assert process_grasp_labels_cdf_width is label_identity
    assert not model.depth_net._forward_hooks
    assert "_select_top_view_inds" not in vars(model.view)
    assert torch.equal(before["torch"], capture_rng_state()["torch"])
    print(f"Directional controller passed: {len(rows)} rows, max frozen FD error {max(errors):.3g}")


if __name__ == "__main__":
    main()
