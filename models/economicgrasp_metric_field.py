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


_OBJECT_PAYLOAD_KEYS = (
    "object_poses_list",
    "grasp_points_list",
    "grasp_rotations_list",
    "grasp_depth_list",
    "grasp_widths_list",
    "grasp_scores_list",
    "view_graspness_list",
    "top_view_index_list",
    "grasp_collision_list",
    "grasp_cdf_bins_list",
    "grasp_widths_depth_list",
    "grasp_width_valids_depth_list",
)


def assert_object_payloads_cpu(end_points):
    """Fail early if DDP/input handling moved variable grasp caches to CUDA."""
    for key in _OBJECT_PAYLOAD_KEYS:
        if key not in end_points:
            continue
        value = end_points[key]
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{key} must be a batch list, got {type(value).__name__}")
        for batch_i, per_sample in enumerate(value):
            if not isinstance(per_sample, (list, tuple)):
                raise TypeError(
                    f"{key}[{batch_i}] must be an object list, "
                    f"got {type(per_sample).__name__}"
                )
            for obj_i, tensor in enumerate(per_sample):
                if torch.is_tensor(tensor) and tensor.device.type != "cpu":
                    raise RuntimeError(
                        f"{key}[{batch_i}][{obj_i}] must remain CPU-resident "
                        f"for online label matching; got device={tensor.device}. "
                        "Do not use DDP(device_ids=[local]) with the full batch "
                        "dict; dense inputs are moved explicitly by move_batch()."
                    )


def filter_empty_grasp_objects(end_points):
    """Drop object slots whose economic-grasp cache contains zero points.

    GraspNet frame metadata can contain an object instance for which the
    scene-level economic-grasp cache has no rows (pointid never equals that
    object slot).  Main's CDF matcher treats this as fatal.  For online MGF
    training we remove that object consistently from *all* aligned object-level
    payloads before matching.  Non-empty objects and every per-object label
    tensor are unchanged.

    A frame in which every object is empty remains an error: silently creating
    all-negative supervision would change the task semantics.
    """
    if "object_poses_list" not in end_points or "grasp_points_list" not in end_points:
        return end_points, []

    poses_batch = end_points["object_poses_list"]
    points_batch = end_points["grasp_points_list"]
    if not isinstance(poses_batch, (list, tuple)) or not isinstance(points_batch, (list, tuple)):
        raise TypeError("object_poses_list and grasp_points_list must be batch lists")
    if len(poses_batch) != len(points_batch):
        raise RuntimeError(
            "object_poses_list/grasp_points_list batch sizes differ: "
            f"{len(poses_batch)} vs {len(points_batch)}"
        )

    out = dict(end_points)
    # Copy outer containers so the dataloader batch is never mutated in-place.
    present_keys = [
        key for key in _OBJECT_PAYLOAD_KEYS
        if key in end_points
    ]
    for key in present_keys:
        value = end_points[key]
        if not isinstance(value, (list, tuple)) or len(value) != len(poses_batch):
            raise TypeError(
                f"{key} must be a batch list of length {len(poses_batch)}, "
                f"got {type(value).__name__}"
            )
        out[key] = list(value)

    reports = []
    for batch_i, (poses, points) in enumerate(zip(poses_batch, points_batch)):
        n_obj = len(poses)
        if len(points) != n_obj:
            raise RuntimeError(
                f"grasp_points_list[{batch_i}] has {len(points)} objects, "
                f"expected {n_obj} from object_poses_list"
            )
        for key in present_keys:
            if len(end_points[key][batch_i]) != n_obj:
                raise RuntimeError(
                    f"{key}[{batch_i}] has {len(end_points[key][batch_i])} "
                    f"objects, expected {n_obj}"
                )

        keep = []
        dropped = []
        for obj_i, pts in enumerate(points):
            if not torch.is_tensor(pts) or pts.dim() != 2 or pts.shape[-1] != 3:
                raise ValueError(
                    f"grasp_points_list[{batch_i}][{obj_i}] must be [P,3], "
                    f"got {type(pts).__name__} "
                    f"{tuple(pts.shape) if torch.is_tensor(pts) else ''}"
                )
            if pts.shape[0] > 0:
                keep.append(obj_i)
            else:
                dropped.append(obj_i)

        if not keep:
            raise RuntimeError(
                f"Batch sample {batch_i} has {n_obj} object labels but all "
                "economic-grasp object caches are empty. Refusing to fabricate "
                "grasp supervision for this frame."
            )

        if dropped:
            for key in present_keys:
                seq = end_points[key][batch_i]
                out[key][batch_i] = [seq[i] for i in keep]
        reports.append(
            {
                "batch_index": batch_i,
                "objects_before": n_obj,
                "objects_after": len(keep),
                "dropped_object_slots": dropped,
            }
        )
    return out, reports


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
        filter_report = []
        if "object_poses_list" in end_points:
            end_points, filter_report = filter_empty_grasp_objects(end_points)
            assert_object_payloads_cpu(end_points)
        end_points["cva_force_process_grasp_labels"] = "object_poses_list" in end_points
        end_points["cva_compute_diagnostics"] = False
        try:
            ep = self.base(end_points)
            dropped = sum(len(row["dropped_object_slots"]) for row in filter_report)
            ep["D: MGF Empty Objects Dropped"] = ep["xyz_graspable"].new_tensor(
                float(dropped)
            ).reshape(())
            ep["mgf_empty_object_filter_report"] = filter_report
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


def _cdf_discriminability_stats(
    logits,
    bins,
    valid,
    *,
    prefix,
    histogram_bins=64,
):
    """Lightweight candidate-level CDF discrimination diagnostics.

    The grasp CDF predicts success at T increasing friction thresholds. Candidate
    utility is the mean threshold success probability. The binary ranking target
    used by AUROC/AUPRC is whether a valid candidate succeeds at ANY configured
    threshold (cdf_bin > 0).

    AUROC/AUPRC use a fixed histogram rather than sorting every candidate. This
    is O(N) and cheap enough to record on every batch. The suffix "64" makes the
    approximation explicit; these values are diagnostics, not benchmark metrics.
    """
    if logits.dim() != 5:
        raise ValueError(
            f"{prefix} CDF logits must be [B,T,Q,A,D], got {tuple(logits.shape)}"
        )
    B, T, Q, A, D = logits.shape
    expected = (B, Q, A, D)
    if bins.shape != expected or valid.shape != expected:
        raise ValueError(
            f"{prefix} CDF labels/mask must be {expected}, got "
            f"bins={tuple(bins.shape)}, valid={tuple(valid.shape)}"
        )
    if histogram_bins < 2:
        raise ValueError("histogram_bins must be >=2")

    with torch.no_grad():
        bins = bins.to(device=logits.device, dtype=torch.long)
        valid = valid.to(device=logits.device, dtype=torch.bool)
        pred_u = torch.sigmoid(logits.float()).mean(dim=1)
        target_u = torch.where(
            bins > 0,
            (float(T) - bins.float() + 1.0) / float(T),
            torch.zeros_like(bins, dtype=torch.float32),
        ).clamp_(0.0, 1.0)
        positive = valid & (bins > 0)
        negative = valid & (~positive)
        zero = pred_u.sum() * 0.0

        if bool(valid.any()):
            pv = pred_u[valid]
            tv = target_u[valid]
            pred_mean = pv.mean()
            target_mean = tv.mean()
            utility_mae = (pv - tv).abs().mean()

            pc = pv - pred_mean
            tc = tv - target_mean
            denom = torch.sqrt(
                pc.square().mean() * tc.square().mean()
            )
            utility_pearson = (
                (pc * tc).mean() / denom.clamp_min(1e-12)
                if bool(denom > 1e-12)
                else zero
            )
        else:
            pred_mean = target_mean = utility_mae = utility_pearson = zero

        pred_pos = pred_u[positive].mean() if bool(positive.any()) else zero
        pred_neg = pred_u[negative].mean() if bool(negative.any()) else zero
        gap = pred_pos - pred_neg
        pos_fraction = (
            positive.float().sum() / valid.float().sum().clamp_min(1.0)
        )

        # Histogram ranking metrics over valid candidates.
        # Scores in [0,1]; index 0=lowest predicted utility.
        if bool(valid.any()):
            score = pred_u[valid].clamp(0.0, 1.0)
            label = positive[valid]
            idx = torch.floor(score * float(histogram_bins)).long()
            idx = idx.clamp_(0, histogram_bins - 1)
            pos_hist = torch.bincount(
                idx[label], minlength=histogram_bins
            ).to(torch.float64)
            neg_hist = torch.bincount(
                idx[~label], minlength=histogram_bins
            ).to(torch.float64)
            P = pos_hist.sum()
            N = neg_hist.sum()

            if bool((P > 0) & (N > 0)):
                neg_before = torch.cumsum(neg_hist, 0) - neg_hist
                # Candidates within one histogram bin are treated as tied.
                auc = (
                    pos_hist * (neg_before + 0.5 * neg_hist)
                ).sum() / (P * N)

                pos_desc = pos_hist.flip(0)
                neg_desc = neg_hist.flip(0)
                tp = torch.cumsum(pos_desc, 0)
                fp = torch.cumsum(neg_desc, 0)
                precision = tp / (tp + fp).clamp_min(1.0)
                recall_increment = pos_desc / P
                auprc = (recall_increment * precision).sum()
            else:
                auc = zero.double()
                auprc = zero.double()
        else:
            auc = zero.double()
            auprc = zero.double()

        lift = (
            auprc.float() / pos_fraction.clamp_min(1e-8)
            if bool(pos_fraction > 0)
            else zero
        )

    return {
        f"{prefix}_cdf_utility_pred_mean": pred_mean.float(),
        f"{prefix}_cdf_utility_target_mean": target_mean.float(),
        f"{prefix}_cdf_utility_mae": utility_mae.float(),
        f"{prefix}_cdf_pred_pos_mean": pred_pos.float(),
        f"{prefix}_cdf_pred_neg_mean": pred_neg.float(),
        f"{prefix}_cdf_pos_neg_gap": gap.float(),
        f"{prefix}_cdf_utility_pearson": utility_pearson.float(),
        f"{prefix}_cdf_positive_fraction": pos_fraction.float(),
        f"{prefix}_cdf_any_success_auroc64": auc.float(),
        f"{prefix}_cdf_any_success_auprc64": auprc.float(),
        f"{prefix}_cdf_any_success_auprc_lift64": lift.float(),
    }


def compute_query_listwise_ranking_loss(
    logits,
    bins,
    valid,
    *,
    temperature=0.1,
    target_range_epsilon=1e-6,
):
    """Within-query listwise ranking over the A x D grasp candidates.

    The existing unbalanced CDF BCE keeps the absolute probability objective.
    This auxiliary loss only asks candidates of the SAME (center, selected-view)
    query to have the correct relative ordering.

    Target utility is the mean binary success over the T friction thresholds:
        cdf_bin=0 -> 0
        cdf_bin=k -> (T-k+1)/T.

    Invalid candidates are excluded. Queries are ranked only when they contain
    at least two valid candidates AND non-constant target utility; all-zero or
    otherwise tied queries carry no ranking information and are skipped.

    The loss is KL(target_softmax || predicted_softmax), using the same
    temperature for target and predicted utilities. Ties in target utility are
    represented naturally by equal target probability.
    """
    if logits.dim() != 5:
        raise ValueError(
            "ranking logits must be [B,T,Q,A,D], got "
            f"{tuple(logits.shape)}"
        )
    B, T, Q, A, D = logits.shape
    expected = (B, Q, A, D)
    if bins.shape != expected or valid.shape != expected:
        raise ValueError(
            "ranking labels/mask must match [B,Q,A,D]="
            f"{expected}, got bins={tuple(bins.shape)}, "
            f"valid={tuple(valid.shape)}"
        )
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            f"ranking temperature must be finite and >0, got {temperature}"
        )
    if target_range_epsilon < 0:
        raise ValueError("target_range_epsilon must be non-negative")

    bins = bins.to(device=logits.device, dtype=torch.long)
    valid = valid.to(device=logits.device, dtype=torch.bool)

    pred_u = torch.sigmoid(logits.float()).mean(dim=1)  # [B,Q,A,D]
    target_u = torch.where(
        bins > 0,
        (float(T) - bins.float() + 1.0) / float(T),
        torch.zeros_like(bins, dtype=torch.float32),
    ).clamp_(0.0, 1.0)

    C = A * D
    pred_flat = pred_u.reshape(B, Q, C)
    target_flat = target_u.reshape(B, Q, C)
    valid_flat = valid.reshape(B, Q, C)
    valid_count = valid_flat.sum(dim=-1)

    target_max = target_flat.masked_fill(~valid_flat, -1.0).max(dim=-1).values
    target_min = target_flat.masked_fill(~valid_flat, 2.0).min(dim=-1).values
    target_range = (target_max - target_min).clamp_min(0.0)
    informative = (
        (valid_count >= 2)
        & (target_range > float(target_range_epsilon))
    )

    # -1e4 is safely negligible at all supported temperatures and avoids
    # infinities in mixed precision / diagnostic computations.
    invalid_logit = pred_flat.new_tensor(-1.0e4)
    pred_rank_logits = (pred_flat / temperature).masked_fill(
        ~valid_flat, invalid_logit
    )
    target_rank_logits = (target_flat / temperature).masked_fill(
        ~valid_flat, invalid_logit
    )
    pred_log_prob = F.log_softmax(pred_rank_logits, dim=-1)
    target_prob = F.softmax(target_rank_logits, dim=-1)
    target_log_prob = torch.log(target_prob.clamp_min(1e-12))

    kl_per_query = (
        target_prob * (target_log_prob - pred_log_prob)
    ).sum(dim=-1)
    zero = logits.sum() * 0.0
    loss = (
        kl_per_query[informative].mean()
        if bool(informative.any())
        else zero
    )

    with torch.no_grad():
        query_total = int(B * Q)
        informative_count = informative.sum()
        informative_fraction = (
            informative.float().mean()
            if query_total > 0 else zero
        )
        range_mean = (
            target_range[informative].mean()
            if bool(informative.any()) else zero
        )

        masked_pred = pred_flat.masked_fill(~valid_flat, -1.0)
        selected_idx = masked_pred.argmax(dim=-1)
        selected_target = target_flat.gather(
            -1, selected_idx.unsqueeze(-1)
        ).squeeze(-1)
        oracle_target = target_max
        selection_regret = (oracle_target - selected_target).clamp_min(0.0)
        top1_best_hit = (
            selected_target >= oracle_target - 1e-7
        )

        selected_target_mean = (
            selected_target[informative].mean()
            if bool(informative.any()) else zero
        )
        oracle_target_mean = (
            oracle_target[informative].mean()
            if bool(informative.any()) else zero
        )
        regret_mean = (
            selection_regret[informative].mean()
            if bool(informative.any()) else zero
        )
        top1_hit = (
            top1_best_hit[informative].float().mean()
            if bool(informative.any()) else zero
        )

    stats = {
        "ranking_loss": loss.detach(),
        "ranking_informative_query_fraction":
            informative_fraction.detach(),
        "ranking_informative_query_count":
            informative_count.detach().float(),
        "ranking_target_range_mean": range_mean.detach(),
        "ranking_selected_target_utility":
            selected_target_mean.detach(),
        "ranking_oracle_target_utility":
            oracle_target_mean.detach(),
        "ranking_selection_regret": regret_mean.detach(),
        "ranking_top1_best_hit": top1_hit.detach(),
    }
    return loss, stats


def metric_field_loss(
    ep,
    config,
    *,
    profile_weight=1.0,
    profile_mean_weight=10.0,
    base_cdf_weight=0.25,
    ranking_weight=1.0,
    ranking_temperature=0.1,
):
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

    cdf_bins = ep["batch_grasp_cdf_bins_angle_depth"].long()
    cdf_valid = ep["batch_grasp_cdf_valid_mask"].bool()
    ranking, ranking_stats = compute_query_listwise_ranking_loss(
        ep["grasp_cdf_pred_angle_depth"],
        cdf_bins,
        cdf_valid,
        temperature=ranking_temperature,
    )
    rank_weight = float(ranking_weight)
    if not math.isfinite(rank_weight) or rank_weight < 0:
        raise ValueError(
            f"ranking_weight must be finite and >=0, got {ranking_weight}"
        )

    task = (
        cfgs.objectness_loss_weight * obj
        + cfgs.graspness_loss_weight * gra
        + cfgs.view_loss_weight * view
        + cfgs.width_loss_weight * width
        + cfgs.score_loss_weight * (
            cdf
            + rank_weight * ranking
            + base_cdf_weight * base_cdf
        )
    )
    geom = geometry_loss(ep["mgf_profile_logits"], ep["depth_map_pred"], ep["gt_depth_m"], config)
    geometric = (cfgs.depth_prob_loss_weight*geom["depth_l1"] +
                 profile_weight*geom["profile_ce"] + profile_mean_weight*geom["profile_mean_l1"])
    total = task + geometric

    # Unit/ranking diagnostics. These are observational only and do not alter
    # gradients. They make catastrophic width-unit mistakes and trivial
    # all-low CDF solutions visible in epoch 0.
    field_discrim = _cdf_discriminability_stats(
        ep["grasp_cdf_pred_angle_depth"],
        cdf_bins,
        cdf_valid,
        prefix="field",
    )
    base_discrim = _cdf_discriminability_stats(
        ep["mgf_base_cdf_logits"],
        cdf_bins,
        cdf_valid,
        prefix="base",
    )

    with torch.no_grad():
        width_label = ep["batch_grasp_width_angle_depth"].float()
        width_valid = ep["batch_grasp_width_valid_mask_angle_depth"].bool()
        pred_width_m = (
            1.2
            * ep["grasp_width_pred_angle_depth"].float().permute(0, 2, 3, 1)
            / 10.0
        ).clamp(0.0, float(cfgs.grasp_max_width))

        zero = total.detach() * 0.0
        width_label_mean = (
            width_label[width_valid].mean()
            if bool(width_valid.any()) else zero
        )
        width_label_max = (
            width_label[width_valid].max()
            if bool(width_valid.any()) else zero
        )
        width_pred_mean = (
            pred_width_m[width_valid].mean()
            if bool(width_valid.any()) else zero
        )
        width_pred_max = (
            pred_width_m[width_valid].max()
            if bool(width_valid.any()) else zero
        )

    stats = {"loss": total, "task_loss": task, "geometry_loss": geometric,
             "cdf": cdf, "base_cdf": base_cdf, "ranking": ranking,
             "ranking_weighted": ranking.detach() * rank_weight,
             "width": width, "objectness": obj,
             "graspness": gra, "view": view,
             **ranking_stats,
             # Backward-compatible alias used by the previous smoke report.
             "cdf_candidate_positive_fraction":
                 field_discrim["field_cdf_positive_fraction"],
             "width_label_m_mean": width_label_mean,
             "width_label_m_max": width_label_max,
             "width_pred_decoded_m_mean": width_pred_mean,
             "width_pred_decoded_m_max": width_pred_max,
             "empty_objects_dropped": ep["D: MGF Empty Objects Dropped"].detach(),
             **field_discrim,
             **base_discrim,
             **geom}
    return total, stats
