"""Online EconomicGrasp-CVA-CDF + DAV2 metric grasp field, based on main.

No P0/P1 feature/action cache. This intentionally keeps main's online nearest-
annotation label assignment; those labels are NOT exact re-evaluations of every
predicted action. No +/-40mm action shifts inherit the native action's label.
"""
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from metric_grasp_field_core import (
    MetricFieldConfig, MetricRayEvidenceHead, TaskFeatureAdapter,
    GraspFieldReadout, geometry_loss,
)

VERSION = "dav2_metric_grasp_field_detach_v1"
DAV2_CONFIGS = {
    "vits": (384, 64, [48, 96, 192, 384]),
    "vitb": (768, 128, [96, 192, 384, 768]),
    "vitl": (1024, 256, [256, 512, 1024, 1024]),
}


def load_trusted_checkpoint(path):
    # Only load the user's trusted local model files. Full checkpoints contain
    # optimizer/RNG metadata, hence the explicit weights_only=False.
    return torch.load(path, map_location="cpu", weights_only=False)


def strip_module(state):
    return {k.removeprefix("module."): v for k, v in state.items()}


def build_candidate_actions(end_points, rotation_fn, max_width=0.1):
    """Exactly match main pred_decode_center_view_angle's physical convention.

    Query-major, angle-major, insertion-depth-minor. Includes the decoder's 1.2
    width expansion, height=.02m and insertion depths .01,...,.04m. NO camera-Z
    offset is added. Labels retain main's <=5mm NN/canonical-angle approximation.
    """
    xyz = end_points["xyz_graspable"].detach().float()
    views = end_points["grasp_top_view_xyz"].detach().float()
    width_dqa = end_points["grasp_width_pred_angle_depth"].detach().float()
    b, d, q, a = width_dqa.shape
    if xyz.shape != (b, q, 3) or views.shape != xyz.shape:
        raise ValueError("CVA query/width shapes disagree")
    width = (1.2 * width_dqa.permute(0, 2, 3, 1) / 10).clamp(0, max_width)
    approaching = -views[:, :, None].expand(-1, -1, a, -1)
    angles = torch.arange(a, device=xyz.device).float() * (torch.pi/a)
    angles = angles[None, None].expand(b, q, -1)
    rotations = rotation_fn(approaching.reshape(-1, 3), angles.reshape(-1))
    rotations = rotations.reshape(b, q, a, 1, 9).expand(-1, -1, -1, d, -1)
    depth = (torch.arange(d, device=xyz.device).float() + 1) * .01
    depth = depth[None, None, None, :, None].expand(b, q, a, -1, -1)
    one = torch.ones_like(width[..., None])
    centre = xyz[:, :, None, None].expand(-1, -1, a, d, -1)
    actions = torch.cat((one, width[..., None], one*.02, depth, rotations, centre, -one), -1)
    return actions.reshape(b, q*a*d, 17), (b, q, a, d)


class EconomicGraspMetricField(nn.Module):
    def __init__(self, config=None, *, encoder="vitb", pose_depth_mode="global_film",
                 init_checkpoint="", seed_selection_mode="image_fps", tok_feat_dim=128):
        super().__init__()
        from .economicgrasp_bip3d import economicgrasp_dpt
        from .dinov2_dpt import DPTHead
        from utils.label_generation import batch_viewpoint_params_to_matrix
        from utils.arguments import cfgs

        self.config = config or MetricFieldConfig()
        self.encoder_name = encoder
        self.pose_depth_mode = pose_depth_mode
        self.seed_selection_mode = seed_selection_mode
        self.rotation_fn = batch_viewpoint_params_to_matrix
        self.max_width = float(cfgs.grasp_max_width)
        if encoder not in DAV2_CONFIGS:
            raise ValueError("Supported encoders: vits/vitb/vitl")
        official_path = Path("checkpoints") / f"depth_anything_v2_{encoder}.pth"
        if not official_path.is_file():
            raise FileNotFoundError(f"Run from repo root; missing official DAV2: {official_path}")
        c = self.config
        self.base = economicgrasp_dpt(
            encoder=encoder, tok_feat_dim=tok_feat_dim, min_depth=c.min_depth,
            max_depth=c.max_depth, freeze_backbone=True, is_training=True,
            use_obs_depth=False, use_depth_comp=False, use_cdf=True,
            pose_depth_mode=pose_depth_mode, seed_selection_mode=seed_selection_mode,
            geometry_depth_source="pred", vis_dir=None,
        )
        if init_checkpoint:
            ck = load_trusted_checkpoint(init_checkpoint)
            if ck.get("geometry_depth_source", "pred") != "pred" or ck.get("use_obs_depth", False):
                raise ValueError("Initialization must be an RGB predicted-depth checkpoint")
            for key, requested in (("pose_depth_mode", pose_depth_mode),
                                   ("seed_selection_mode", seed_selection_mode)):
                if key in ck and ck[key] != requested:
                    raise ValueError(f"Init {key}={ck[key]!r}, requested {requested!r}")
            self.base.load_state_dict(strip_module(ck.get("model_state_dict", ck)), strict=True)
            del ck

        # Restore the ORIGINAL relative decoder, not a copy of the task-trained
        # metric DPT. Encoder equality is verified before sharing its forward.
        official = load_trusted_checkpoint(official_path)
        official = strip_module(official.get("model_state_dict", official))
        encoder_state = self.base.depth_net.depthnet.pretrained.state_dict()
        for key, value in encoder_state.items():
            expected = official.get("pretrained." + key)
            if expected is None or not torch.equal(value.cpu(), expected.cpu()):
                raise RuntimeError(f"Shared encoder differs from official DAV2 at {key}; refusing silent reuse")
        embed, geom_dim, channels = DAV2_CONFIGS[encoder]
        self.relative_decoder = DPTHead(embed, features=geom_dim, out_channels=channels,
                                        use_clstoken=False, out_dim=1)
        rel_state = {k[len("depth_head."):]: v for k, v in official.items() if k.startswith("depth_head.")}
        self.relative_decoder.load_state_dict(rel_state, strict=True)
        self.relative_decoder.requires_grad_(False).eval()
        del official, rel_state
        self.ray_head = MetricRayEvidenceHead(geom_dim, geom_dim, c)
        self.task_adapter = TaskFeatureAdapter(tok_feat_dim, geom_dim, geom_dim, c.hidden)
        self.readout = GraspFieldReadout(c)
        self._depth_pack = None
        self._proposal_feature = None
        # Hooks avoid changing main's 10k-line model or state-dict layout.
        self.base.depth_net.register_forward_hook(self._depth_boundary)
        self.base.proposal_head.register_forward_hook(self._capture_proposal)

    def _depth_boundary(self, module, inputs, outputs):
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
            raise RuntimeError("main metric-depth tuple contract changed (expected six outputs)")
        if self._depth_pack is not None:
            raise RuntimeError("Expected exactly one depth/encoder forward per input batch")
        self._depth_pack = outputs
        result = list(outputs)
        # Preserve online depth supervision in _depth_pack, but block every
        # numeric depth/metric-feature path into main's proposal/view/CVA graph.
        for i in (0, 1, 2, 3):
            result[i] = result[i].detach()
        return tuple(result)

    def _capture_proposal(self, module, inputs, outputs):
        self._proposal_feature = outputs[0]

    def train(self, mode=True):
        super().train(mode)
        self.relative_decoder.eval()
        self.base.depth_net.depthnet.pretrained.eval()
        # main uses this explicit flag for view sampling and label processing.
        self.base.is_training = bool(mode)
        self.base.view.is_training = bool(mode)
        return self

    def geometry_parameters(self):
        return [p for m in (self.base.depth_net, self.ray_head)
                for p in m.parameters() if p.requires_grad]

    def forward(self, batch):
        self._depth_pack = self._proposal_feature = None
        # Validation gets labels but deterministic eval proposals; inference
        # supplies no annotation payload, so no GT-dependent prediction branch.
        end_points = dict(batch)
        end_points["cva_force_process_grasp_labels"] = "object_poses_list" in batch
        end_points["cva_compute_diagnostics"] = False
        try:
            ep = self.base(end_points)
            if self._depth_pack is None or self._proposal_feature is None:
                raise RuntimeError("Required base feature hooks were not executed")
            depth, _, metric_feat, raw, feats, _ = self._depth_pack
            h, w = batch["img"].shape[-2:]
            with torch.no_grad():
                relative_feat, relative_raw = self.relative_decoder(feats, h//14, w//14)
                relative_inverse = F.relu(relative_raw)
            logits, prob = self.ray_head(relative_feat, relative_inverse, metric_feat, depth, (h, w), batch["K"])
            task_feature = self.task_adapter(self._proposal_feature, relative_feat, metric_feat,
                                             logits.shape[-2:])
            actions, (b, q, a, d) = build_candidate_actions(ep, self.rotation_fn, self.max_width)
            field_logits = self.readout(task_feature, prob.detach(), actions, batch["K"], (h, w))
            ep["mgf_base_cdf_logits"] = ep["grasp_cdf_pred_angle_depth"]
            ep["grasp_cdf_pred_angle_depth"] = field_logits.reshape(b, q, a, d, 6).permute(0, 4, 1, 2, 3).contiguous()
            ep["mgf_profile_logits"] = logits
            # Original undetached prediction for metric-only supervision.
            ep["depth_map_pred"] = depth
            ep["depth_net_pred"] = depth
            ep["depth_head_raw_pred"] = raw
            ep["mgf_geometry_depth"] = depth.detach()
            ep["mgf_profile_mean"] = (prob * self.ray_head.centres[None, :, None, None]).sum(1).detach()
            return ep
        finally:
            self._depth_pack = self._proposal_feature = None


def metric_field_loss(ep, config, *, profile_weight=1., profile_mean_weight=10.,
                      base_cdf_weight=.25):
    from .loss_economicgrasp_depth_kview_transformer import (
        compute_objectness_loss_tok, compute_graspness_loss_tok,
        compute_view_graspness_loss, compute_cva_cdf_loss, compute_cva_width_depth_loss,
    )
    from utils.arguments import cfgs
    obj, _ = compute_objectness_loss_tok(ep)
    gra, _ = compute_graspness_loss_tok(ep)
    view, _ = compute_view_graspness_loss(ep)
    cdf, _ = compute_cva_cdf_loss(ep, balanced=False)
    width, _ = compute_cva_width_depth_loss(ep)
    auxiliary = dict(ep)
    auxiliary["grasp_cdf_pred_angle_depth"] = ep["mgf_base_cdf_logits"]
    base_cdf, _ = compute_cva_cdf_loss(auxiliary, balanced=False)
    task = (cfgs.objectness_loss_weight*obj + cfgs.graspness_loss_weight*gra +
            cfgs.view_loss_weight*view + cfgs.width_loss_weight*width +
            cfgs.score_loss_weight*(cdf + base_cdf_weight*base_cdf))
    geom = geometry_loss(ep["mgf_profile_logits"], ep["depth_map_pred"], ep["gt_depth_m"], config)
    geometric = (cfgs.depth_prob_loss_weight*geom["depth_l1"] +
                 profile_weight*geom["profile_ce"] + profile_mean_weight*geom["profile_mean_l1"])
    total = task + geometric
    stats = {"loss": total, "task_loss": task, "geometry_loss": geometric,
             "cdf": cdf, "base_cdf": base_cdf, "width": width, "objectness": obj,
             "graspness": gra, "view": view, **geom}
    return total, stats
