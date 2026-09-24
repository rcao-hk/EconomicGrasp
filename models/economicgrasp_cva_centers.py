"""Online-RGB E1/E2 integration with an immutable Stage-1 action generator.

The trainable path is DPT image adapter -> spatial enhancer -> center/angle
conditioned CVA grouping -> existing monotonic CDF decoder. The frozen Stage-1
provides depth and native R/w/d so exact physical-action labels remain valid.
E1/E2 are identical networks; their only difference is the training objective.
No Rep-A/B/C module or cached image-feature map is used.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from e1e2_common import expand_centers, load_torch, perturb_depth


def load_reference(checkpoint, device):
    # Call ONLY after the command-line parser has isolated legacy sys.argv.
    from utils.arguments import cfgs
    from models.economicgrasp_dpt_distill import economicgrasp_dpt_student
    ck = load_torch(checkpoint)
    if ck.get('geometry_depth_source', 'pred') != 'pred' or ck.get('use_obs_depth', False):
        raise ValueError('An RGB-only predicted-depth Stage-1 checkpoint is required')
    pose = ck.get('pose_depth_mode', 'global_film')
    if pose != 'global_film':
        raise ValueError(f'This initial E1/E2 protocol requires global_film, got {pose}')
    cfgs.use_top4_view_infer = False; cfgs.kview_mode = 'A1'; cfgs.kview_k = 1
    cfgs.use_cdf = True; cfgs.use_obs_depth = False; cfgs.pose_depth_mode = pose
    for name, default in [('num_angle', 12), ('num_depth', 4), ('num_view', 300),
                          ('m_point', 1024), ('graspness_threshold', .1), ('grasp_max_width', .1)]:
        # Keep main defaults unless the checkpoint explicitly records a value.
        setattr(cfgs, name, ck.get(name, getattr(cfgs, name, default)))
    kwargs = {k: ck.get(k, v) for k, v in [('camera_pose_key', 'camera_pose_vec'),
              ('camera_gravity_key', 'camera_gravity_vec'), ('pose_hidden_dim', 64),
              ('ray_gravity_hidden_dim', 64), ('ray_gravity_mid_dim', 32)]}
    model = economicgrasp_dpt_student(
        encoder=ck.get('encoder', 'vitb'), is_training=False, use_cdf=True,
        use_obs_depth=False, pose_depth_mode=pose, freeze_backbone=True,
        min_depth=.2, max_depth=1., bin_num=256, vis_dir=None, **kwargs).to(device)
    state = ck['model_state_dict']
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    del ck
    return model


@contextmanager
def _depth_override(depth_net, output):
    # Scoped, restored even on exception; only immutable reference uses this.
    present = 'forward' in depth_net.__dict__
    old = depth_net.__dict__.get('forward')
    depth_net.forward = lambda *a, **kw: output
    try:
        yield
    finally:
        if present:
            depth_net.forward = old
        else:
            del depth_net.forward


@torch.no_grad()
def extract_depth_features(reference, batch):
    pack = reference.depth_net(
        batch['img'], camera_pose_vec=batch[reference.camera_pose_key],
        camera_gravity_vec=None, camera_K=batch['K'], return_feats=True,
        return_raw=True, return_pose_aux=True)
    if not isinstance(pack, (tuple, list)) or len(pack) != 6:
        raise RuntimeError('Main depth_net contract changed: expected six outputs')
    return pack


@torch.no_grad()
def reference_candidates(reference, batch, pack, case, seed, offsets, query_limit=0):
    """True joint replay: the perturbed depth reaches enhancer, seeds, ViewNet/CVA."""
    from models.economicgrasp_bip3d import pred_decode_center_view_angle, _cva_decode_query_indices
    active_depth = perturb_depth(pack[0], case, seed)
    with _depth_override(reference.depth_net, (active_depth, *pack[1:])):
        ep = reference(dict(batch, cva_compute_diagnostics=False, cva_export_angle_feature=False,
                            geometry_compute_diagnostics=False))
    if not torch.equal(ep['depth_map_used_for_geometry'], active_depth):
        raise RuntimeError('Depth perturbation failed to reach Stage-1 geometry')
    native_all = pred_decode_center_view_angle(ep, use_cdf=True)[0]
    total = len(native_all)
    idx = _cva_decode_query_indices(ep, 0, ep['xyz_graspable'].shape[1], False)
    if len(idx) != total:
        raise RuntimeError('CVA decode/query alignment changed')
    # Deterministic uniform query coverage, no label/high-margin based sampling.
    keep = (torch.arange(total, device=idx.device) if query_limit == 0 or query_limit >= total
            else torch.linspace(0, total-1, query_limit, device=idx.device).round().long())
    qidx = idx[keep]
    native = native_all[keep]
    utility = ep['grasp_cdf_pred_angle_depth'][0].sigmoid().mean(0)[qidx]
    nd = utility.shape[-1]
    ad = utility.flatten(1).argmax(1)
    angle_ids, depth_ids = ad // nd, ad % nd
    if not torch.allclose(native[:, 3], (depth_ids+1).float()*.01, atol=1e-6):
        raise RuntimeError('Native insertion-depth/index alignment failed')
    actions, valid = expand_centers(native, offsets, reference.min_depth, reference.max_depth)
    zero = int(np.flatnonzero(np.asarray(offsets) == 0)[0])
    if not bool(valid[zero].all()):
        raise RuntimeError('Stage-1 produced out-of-range native centers')
    # Exact base token mapping, not image projection rounded back to a pixel.
    token_ids = ep['token_sel_idx'][0][qidx]
    return {
        'actions': actions, 'valid': valid, 'native': native,
        'token_ids': token_ids, 'view_xyz': ep['grasp_top_view_xyz'][0][qidx],
        'angle_ids': angle_ids, 'depth_ids': depth_ids, 'query_ids': qidx,
        'zero': zero, 'total_stage1_queries': total,
    }, active_depth


class CenterHypothesisCVA(nn.Module):
    """Fixed-action stage of end-to-end representation training; one frame/call."""
    def __init__(self, reference, offsets_mm, group_chunk=512):
        super().__init__()
        self.reference = reference.eval().requires_grad_(False)
        self.register_buffer('offsets_mm', torch.as_tensor(offsets_mm, dtype=torch.float32))
        self.zero = int(torch.nonzero(self.offsets_mm == 0).reshape(-1).item())
        self.image_adapter = copy.deepcopy(reference.proposal_head).requires_grad_(True)
        self.enhancer = copy.deepcopy(reference.spatial_enhancer).requires_grad_(True)
        self.group = copy.deepcopy(reference.kview_grasp_module.group).requires_grad_(True)
        self.decoder = copy.deepcopy(reference.kview_grasp_module.decoder).requires_grad_(True)
        self.group.config = copy.deepcopy(self.group.config)
        self.group.config.grouping_max_queries_per_chunk = int(group_chunk)
        self.group.config.detach_depth = True
        self.group.config.detach_aux_maps = True
        self.group.config.vis_dir = None
        self.num_angle = int(reference.num_angle)
        self.num_depth = int(reference.num_depth)
        dim = int(reference.kview_config.head_model_dim)
        # Complete action context (including frozen w/d); zero residual at init.
        self.action_adapter = nn.Sequential(nn.Linear(15, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.action_adapter[-1].weight); nn.init.zeros_(self.action_adapter[-1].bias)
        # Width is never changed by this initial translation-only protocol.
        self.decoder.width_head.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.reference.eval()
        return self

    def learned_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith('reference.')}

    def load_learned_state(self, state):
        expected = {k for k in self.state_dict() if not k.startswith('reference.')}
        if set(state) != expected:
            raise RuntimeError(f'Learned state mismatch: missing={expected-set(state)}, extra={set(state)-expected}')
        if not torch.equal(state['offsets_mm'].cpu(), self.offsets_mm.cpu()):
            raise RuntimeError('Checkpoint center-offset grid mismatch')
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith('reference.') for k in missing):
            raise RuntimeError('Invalid learned checkpoint')

    def encode_image(self, batch, pack, active_depth):
        """Frozen backbone evaluated online; DPT adapter gets grasp gradients."""
        h, w = batch['img'].shape[-2:]
        raw, proposal_logits = self.image_adapter(pack[4], h//14, w//14)
        enhanced, _ = self.enhancer(raw, depth_prob=None, depth_map=active_depth.detach(),
                                    K=batch['K'], image_hw=(h, w), return_maps=False)
        feature = F.interpolate(enhanced, size=(h, w), mode='bilinear', align_corners=False)
        return feature, proposal_logits

    def score_bundle(self, feature, proposal_logits, batch, active_depth, bundle, return_features=False):
        from utils.label_generation import batch_viewpoint_params_to_matrix
        actions, valid = bundle['actions'], bundle['valid']
        c, q = valid.shape
        a = self.num_angle
        # Invalid locations are never evaluated/decoded. Supply finite geometry
        # to attention and mask their loss, rather than allowing NaNs to spread.
        safe = torch.where(valid[..., None], actions, actions[self.zero:self.zero+1])
        xyz = safe[..., 13:16].unsqueeze(2).expand(c, q, a, 3).reshape(1, c*q*a, 3)
        token = bundle['token_ids'].view(1, q, 1).expand(c, q, a).reshape(1, -1)
        view = bundle['view_xyz'].view(1, q, 1, 3).expand(c, q, a, 3).reshape(-1, 3)
        angles = torch.arange(a, device=view.device, dtype=view.dtype).view(1, 1, a).expand(c, q, a)
        rotations = batch_viewpoint_params_to_matrix(-view, angles.reshape(-1)*(np.pi/a)).reshape(1, -1, 3, 3)
        seeds = feature.flatten(2).gather(2, token[:, None].expand(1, feature.shape[1], -1))
        grouped = self.group(
            seed_features=seeds, token_sel_idx=token, seed_xyz=xyz, top_view_rot=rotations,
            feat_map=feature, depth_map=active_depth.detach(),
            objectness_logits=proposal_logits[:, :2], graspness_map=proposal_logits[:, 2:3],
            camera_K=batch['K'], end_points={})
        complete_action = safe[:, :, None].expand(c, q, a, 17).clone().reshape(-1, 17)
        complete_action[:, 4:13] = rotations.reshape(-1, 9)
        grouped = grouped + self.action_adapter(complete_action[:, 1:16]).T[None]
        ep = self.decoder(grouped, {'kview_angle_query_base_q': c*q, 'kview_angle_query_num_angle': a})
        grid = ep['grasp_cdf_pred_angle_depth'][0].permute(1, 2, 3, 0)
        qi = torch.arange(c*q, device=grid.device)
        ai = bundle['angle_ids'].repeat(c)
        di = bundle['depth_ids'].repeat(c)
        # Fixed R/w/insertion depth. Other A,D outputs are not given stale labels.
        logits = grid[qi, ai, di].reshape(c, q, 6)
        if return_features:
            # Latent for this exact native angle and translated center. No
            # change to old logits, grouping, decoder state or action labels.
            latent = grouped[0].T.reshape(c*q, a, -1)[qi, ai].reshape(c, q, -1)
            return logits, latent
        return logits

    def forward(self, batch, bundle=None, case='nominal', case_seed=0, query_limit=0, query_chunk=64, depth_pack=None, return_features=False):
        if batch['img'].shape[0] != 1:
            raise ValueError('Use one frame per call; gradient accumulation controls effective batch size')
        if query_chunk < 1:
            raise ValueError('query_chunk must be positive')
        pack = extract_depth_features(self.reference, batch) if depth_pack is None else depth_pack
        active = perturb_depth(pack[0], case, case_seed)
        if bundle is None:
            bundle, active = reference_candidates(self.reference, batch, pack, case, case_seed,
                                                  self.offsets_mm.cpu().numpy(), query_limit)
        feature, proposal_logits = self.encode_image(batch, pack, active)
        n = bundle['actions'].shape[1]
        outputs, latents = [], []
        for start in range(0, n, query_chunk):
            end = min(n, start+query_chunk)
            sub = {k: (v[:, start:end] if k in ('actions', 'valid') else
                       v[start:end] if k in ('token_ids', 'view_xyz', 'angle_ids', 'depth_ids') else v)
                   for k, v in bundle.items()}
            result = self.score_bundle(feature, proposal_logits, batch, active, sub, return_features)
            if return_features:
                logits, latent = result
                outputs.append(logits); latents.append(latent)
            else:
                outputs.append(result)
        result = (torch.cat(outputs, 1), bundle, pack[0])
        return (*result, torch.cat(latents, 1)) if return_features else result
