"""Audited main-runtime adapter: own image features, common predicted geometry.

No foreign hidden features are exchanged. Replay recomputes spatial enhancer,
seed feature gathering, ViewNet, RayRoPE and GVAR at the reference queries.
The normal model code is unchanged; temporary method/hook changes are restored.
Historical models need an explicit adapter with the same contract, not relaxed
state-dict loading. The public adapter class can wrap a compatible old model.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import importlib
import inspect
import sys
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F

from .contracts import digest_array, file_sha256, validate_grasps
from .hooks import reader_intervention


def detached(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {k: detached(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(detached(v) for v in value)
    return value


def tensor_hash(x):
    return digest_array(x.detach().float().cpu().numpy())


def assert_close(a, b, label, atol=1e-5):
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError(f'{label}: shape or finiteness mismatch')
    error = float((a.float() - b.float()).abs().max()) if a.numel() else 0.
    if error > atol:
        raise RuntimeError(f'{label}: max absolute difference {error:.9g} > {atol}')
    return error


@dataclass
class Snapshot:
    spatial: dict
    kview: dict
    native: dict
    image_hash: str
    input_hashes: dict
    checkpoint_sha: str
    settings: dict

    @property
    def depth(self):
        return self.kview['depth_map']

    @property
    def geometry_hash(self):
        parts = [tensor_hash(self.depth), tensor_hash(self.kview['camera_K'])]
        for key in ('xyz_graspable', 'token_sel_idx', 'grasp_top_view_inds', 'grasp_top_view_xyz'):
            parts.append(tensor_hash(self.native[key]))
        import hashlib
        return hashlib.sha256('|'.join(parts).encode()).hexdigest()


class MainRuntime:
    """Requires compatible model modules, not a particular checkpoint filename."""
    def __init__(self, model, checkpoint_sha, settings, activation=None, decode_fn=None):
        self.model = model.eval()
        self.checkpoint_sha = checkpoint_sha
        self.settings = dict(settings)
        if not np.isclose(float(self.settings['max_width']), .1, atol=1e-12, rtol=0):
            raise ValueError('This fixed P3-compatible experiment requires the audited 0.10m width cap')
        self.activation = activation or nullcontext
        self.decode_fn = decode_fn
        self.cva = model.kview_grasp_module
        self.reader = getattr(self.cva.decoder, 'reader', None)
        if not getattr(model, 'enable_point_rayrope', False):
            raise ValueError('Common-bank experiment requires Point RayRoPE grouping')
        if getattr(model, 'use_obs_depth', False) or getattr(model, 'use_gt_xyz_for_train', False):
            raise ValueError('Observed/GT depth is not permitted in the model path')
        if getattr(model, 'geometry_depth_source', 'pred') != 'pred':
            raise ValueError('Reference geometry must be RGB-predicted depth')
        if str(self.cva.config.mode).upper() != 'A1' or not model.use_cdf:
            raise ValueError('Only the audited single-top-view A1/CDF contract is supported')
        self.device = next(model.parameters()).device

    def capture(self, batch):
        # Labels, sensor point clouds, segmentation and observed/GT depth are
        # not passed to the network. Dataset crop/workspace may still use depth.
        allowed = {'img', 'K', 'camera_pose_vec', 'camera_gravity_vec',
                   'img_idxs', 'token_valid_mask', 'camera_pose', 'scene_idx', 'anno_idx'}
        for name in ('camera_pose_key', 'camera_gravity_key'):
            if self.settings.get(name):
                allowed.add(self.settings[name])
        inputs = {k: v.to(self.device) for k, v in batch.items()
                  if k in allowed and torch.is_tensor(v)}
        if 'img' not in inputs or 'K' not in inputs or inputs['img'].shape[0] != 1:
            raise ValueError('capture requires a single RGB/K frame')
        inputs.update(cva_export_angle_feature=False, cva_compute_diagnostics=False,
                      geometry_compute_diagnostics=False, cva_force_process_grasp_labels=False)
        cache = {}
        def spatial_hook(_module, args, kwargs):
            if args or 'depth_map' not in kwargs or kwargs.get('depth_prob') is not None:
                raise RuntimeError('Unsupported spatial enhancer call contract')
            if 'spatial' in cache:
                raise RuntimeError('Multiple spatial enhancer calls; explicit adapter required')
            cache['spatial'] = detached(dict(kwargs))
        def cva_hook(_module, args, kwargs):
            if args or kwargs.get('is_training') or kwargs.get('process_grasp_labels_fn'):
                raise RuntimeError('Expected keyword-only, label-free inference CVA call')
            if 'kview' in cache:
                raise RuntimeError('Multiple CVA calls; explicit adapter required')
            cache['kview'] = detached({k: v for k, v in kwargs.items() if k != 'end_points'})
        handles = [self.model.spatial_enhancer.register_forward_pre_hook(spatial_hook, with_kwargs=True),
                   self.cva.register_forward_pre_hook(cva_hook, with_kwargs=True)]
        try:
            with self.activation(), torch.inference_mode():
                output = self.model(dict(inputs))
        finally:
            for h in handles: h.remove()
        required = ('xyz_graspable', 'token_sel_idx', 'grasp_top_view_inds',
                    'grasp_top_view_xyz', 'grasp_cdf_pred_angle_depth',
                    'grasp_width_pred_angle_depth')
        if set(cache) != {'spatial', 'kview'} or any(k not in output for k in required):
            raise RuntimeError('Incomplete model capture')
        native = {k: output[k].detach().clone() for k in required}
        if self.decode_fn is not None:
            with self.activation(), torch.inference_mode():
                decoded = self.decode_fn(output, use_cdf=True)[0].detach().clone()
            if decoded.ndim != 2 or decoded.shape != (native['xyz_graspable'].shape[1], 17):
                raise RuntimeError('Production decoder did not return one native winner per query')
            assert_close(decoded[:, 13:16], native['xyz_graspable'][0], 'production decoder query order', 0)
            native['decoded_native'] = decoded
        cdf = native['grasp_cdf_pred_angle_depth']
        if cdf.ndim != 5 or cdf.shape[0] != 1 or cdf.shape[1] != 6 or cdf.shape[-1] != 4:
            raise RuntimeError('Expected CDF [1,6,Q,A,4]')
        if cache['kview']['depth_map'].shape[-2:] != inputs['img'].shape[-2:]:
            raise RuntimeError('Depth and resized RGB must share resolution')
        assert_close(cache['spatial']['depth_map'], cache['kview']['depth_map'], 'native depth routing', 0)
        return Snapshot(cache['spatial'], cache['kview'], native,
                        tensor_hash(inputs['img']),
                        {k: tensor_hash(v) for k, v in inputs.items() if torch.is_tensor(v)},
                        self.checkpoint_sha, self.settings)

    def replay(self, own: Snapshot, reference: Snapshot, condition='native'):
        if own.image_hash != reference.image_hash or own.input_hashes != reference.input_hashes:
            raise ValueError('Cross-model RGB/K/pose/token input preprocessing differs')
        if own.settings['max_width'] != reference.settings['max_width']:
            raise ValueError('Models must declare the same executed gripper width cap')
        if own.native['grasp_cdf_pred_angle_depth'].shape != reference.native['grasp_cdf_pred_angle_depth'].shape:
            raise ValueError('Different query/angle/depth lattices; no silent nearest-neighbor matching')
        ref = reference.native
        depth, K = reference.depth, reference.kview['camera_K']
        indices, centers = ref['token_sel_idx'], ref['xyz_graspable']
        views = ref['grasp_top_view_inds']
        if depth.device != self.device:
            raise ValueError('Load both diagnostic runtimes on the same device')
        if torch.is_grad_enabled():
            # Public entrypoint remains safe outside a torch context.
            with torch.inference_mode():
                return self.replay(own, reference, condition)
        selector = self.cva.selector
        old_select = selector.__dict__.get('_select_view_indices')
        had_select = '_select_view_indices' in selector.__dict__
        original_view = self.cva._call_view_net
        old_view = self.cva.__dict__.get('_call_view_net')
        had_view = '_call_view_net' in self.cva.__dict__
        counts = dict(spatial=0, view=0, group=0, reader=0)
        def fixed_select(_selector, view_score, is_training, forced_view_inds=None):
            if is_training or view_score.shape[:2] != views.shape:
                raise RuntimeError('Fixed view selection received mismatched queries')
            if torch.any((views < 0) | (views >= view_score.shape[-1])):
                raise ValueError('Reference view outside target canonical view grid')
            prob = _selector._make_prob(view_score)
            idx = views.unsqueeze(-1)
            return idx, torch.zeros_like(idx), prob.gather(-1, idx), 1, prob
        def checked_view(_self, **kwargs):
            assert_close(kwargs['depth_map'], depth, 'ViewNet common geometry', 0)
            counts['view'] += 1
            return original_view(**kwargs)
        def check_group(_module, args, kwargs):
            assert_close(kwargs['depth_map'], depth, 'RayRoPE common geometry', 0)
            counts['group'] += 1
        def check_reader(_module, args):
            assert_close(args[1]['depth'], depth, 'GVAR reference geometry before local intervention', 0)
            counts['reader'] += 1
        handles = [self.cva.group.register_forward_pre_hook(check_group, with_kwargs=True)]
        if self.reader is not None:
            handles.append(self.reader.register_forward_pre_hook(check_reader))
        selector._select_view_indices = MethodType(fixed_select, selector)
        self.cva._call_view_net = MethodType(checked_view, self.cva)
        try:
            with self.activation(), reader_intervention(self.reader, condition):
                spatial = dict(own.spatial)
                spatial.update(depth_map=depth, depth_prob=None, K=K,
                               image_hw=tuple(depth.shape[-2:]))
                enhanced, _ = self.model.spatial_enhancer(**spatial)
                counts['spatial'] += 1
                feat_map = F.interpolate(enhanced, size=depth.shape[-2:], mode='bilinear', align_corners=False)
                b, c, h, w = feat_map.shape
                flat = feat_map.reshape(b, c, h * w)
                if torch.any((indices < 0) | (indices >= h * w)):
                    raise ValueError('Reference pixel seed indices outside target grid')
                seed_features = flat.gather(2, indices[:, None].expand(-1, c, -1)).contiguous()
                endpoints = dict(cva_export_angle_feature=False, cva_compute_diagnostics=False,
                                 geometry_compute_diagnostics=False, img=own.kview.get('img'), K=K,
                                 xyz_graspable=centers, token_sel_idx=indices, token_sel_xyz=centers)
                if self.reader is not None:
                    endpoints['_gvar_pregeom'] = own.spatial['feat_2d']
                kwargs = dict(own.kview)
                kwargs.update(seed_features=seed_features, seed_xyz=centers, token_sel_idx=indices,
                              feat_map=feat_map, depth_map=depth, depth_prob=None,
                              camera_K=K, end_points=endpoints, is_training=False,
                              process_grasp_labels_fn=None, process_grasp_labels_kwargs=None,
                              topview_debug_fn=None)
                out = self.cva(**kwargs)
        finally:
            for handle in handles: handle.remove()
            if had_select: selector._select_view_indices = old_select
            else: del selector._select_view_indices
            if had_view: self.cva._call_view_net = old_view
            else: del self.cva._call_view_net
        assert_close(out['xyz_graspable'], centers, 'fixed physical centers', 0)
        assert_close(out['grasp_top_view_xyz'], ref['grasp_top_view_xyz'], 'same canonical view rays', 1e-6)
        if not torch.equal(out['token_sel_idx'], indices):
            raise RuntimeError('Seed pixel identities changed')
        if counts != dict(spatial=1, view=1, group=1, reader=int(self.reader is not None)):
            raise RuntimeError(f'Unverified geometry routes: {counts}')
        result = {k: out[k].detach() for k in ('grasp_cdf_pred_angle_depth', 'grasp_width_pred_angle_depth')}
        if any(not torch.isfinite(v).all() for v in result.values()):
            raise FloatingPointError('Nonfinite replay prediction')
        return result, dict(routes=counts, common_geometry_sha256=reference.geometry_hash,
                            own_checkpoint_sha256=self.checkpoint_sha,
                            condition=condition, own_features_not_exchanged=True)

    def validate_native(self, snapshot, atol=1e-5):
        prediction, audit = self.replay(snapshot, snapshot)
        audit['native_max_abs_errors'] = {
            k: assert_close(prediction[k], snapshot.native[k], f'native replay {k}', atol)
            for k in prediction}
        if self.reader is not None:
            identity, _ = self.replay(snapshot, snapshot, 'perm_identity')
            audit['identity_permutation_max_abs_errors'] = {
                k: assert_close(identity[k], prediction[k], f'identity residual {k}', 0)
                for k in prediction}
        return audit

    def make_bank(self, snapshot, n_families, seed):
        cdf = snapshot.native['grasp_cdf_pred_angle_depth'][0].float().sigmoid().mean(0)
        q, a, d = cdf.shape
        if d != 4 or not 1 <= n_families <= q:
            raise ValueError('Invalid query-family budget')
        ids = np.sort(np.random.default_rng(seed).choice(q, n_families, replace=False))
        qs = torch.as_tensor(ids, device=cdf.device, dtype=torch.long)
        winners = cdf.reshape(q, -1).argmax(-1)
        angle_ids = (winners // d).index_select(0, qs)
        views = snapshot.native['grasp_top_view_xyz'][0].index_select(0, qs)
        angles = angle_ids.float() * (np.pi / a)
        rotation = self.cva.selector.batch_viewpoint_params_to_matrix_fn(-views, angles)
        if 'decoded_native' in snapshot.native:
            # Use production rotations byte-for-byte; avoid alternate angle
            # floating-point evaluation order in the reconstructed bank.
            production = snapshot.native['decoded_native'].index_select(0, qs)
            rotation = production[:, 4:13].reshape(-1, 3, 3)
        centers = snapshot.native['xyz_graspable'][0].index_select(0, qs)
        width = snapshot.native['grasp_width_pred_angle_depth'][0].float().permute(1, 2, 0)
        width = (1.2 * width / 10).clamp(0, snapshot.settings['max_width'])
        bank = torch.zeros((n_families, d, 17), device=cdf.device, dtype=torch.float32)
        bank[..., 0] = cdf[qs, angle_ids]
        bank[..., 1] = width[qs, angle_ids]
        bank[..., 2] = .02
        bank[..., 3] = torch.arange(1, 5, device=cdf.device).float() * .01
        bank[..., 4:13] = rotation.reshape(n_families, 1, 9)
        bank[..., 13:16] = centers[:, None]
        bank[..., 16] = -1
        if 'decoded_native' in snapshot.native:
            chosen = (winners % d).index_select(0, qs)
            selected = bank[torch.arange(n_families, device=cdf.device), chosen]
            assert_close(selected, production, 'native bank vs production 17D decoder', 1e-6)
        bank = bank.cpu().numpy()
        validate_grasps(bank, snapshot.settings['max_width'])
        return bank, ids, angle_ids.cpu().numpy()

    def selected_outputs(self, predictions, query_ids, angle_ids):
        scores = predictions['grasp_cdf_pred_angle_depth'][0].float().sigmoid().mean(0)
        q = torch.as_tensor(query_ids, device=scores.device)
        a = torch.as_tensor(angle_ids, device=scores.device)
        widths = predictions['grasp_width_pred_angle_depth'][0].float().permute(1, 2, 0)
        widths = (widths * 1.2 / 10).clamp(0, self.settings['max_width'])
        return scores[q, a].cpu().numpy(), widths[q, a].cpu().numpy()


def build_main_runtime(config: dict, device: str):
    """config: checkpoint, expected_sha256, model_args (the exact training CLI).

    utils.arguments is legacy import-time argparse. Temporarily isolate argv and
    restore the *same* cfgs object around every model call; no parallel threads.
    """
    checkpoint = config['checkpoint']
    actual_sha = file_sha256(checkpoint)
    if actual_sha != config.get('expected_sha256'):
        raise ValueError('Checkpoint SHA256 is required and must match exactly')
    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        arguments = importlib.import_module('utils.arguments')
    finally:
        sys.argv = argv
    namespace = arguments.parser.parse_args(config.get('model_args', []))
    saved = dict(vars(namespace))
    saved['checkpoint_path'] = checkpoint
    saved['vis_dir'] = None
    if not saved.get('multi_modal') or not saved.get('use_cdf') or not saved.get('enable_point_rayrope'):
        raise ValueError('Pass --multi_modal --use_cdf --enable_point_rayrope in model_args')
    if saved.get('use_obs_depth') or saved.get('use_gt_depth') or saved.get('use_top4_view_infer'):
        raise ValueError('Expected RGB predicted geometry and a single native view')
    @contextmanager
    def activation():
        old = dict(vars(arguments.cfgs))
        arguments.cfgs.__dict__.clear(); arguments.cfgs.__dict__.update(saved)
        try: yield
        finally:
            arguments.cfgs.__dict__.clear(); arguments.cfgs.__dict__.update(old)
    with activation():
        from models.economicgrasp_bip3d import economicgrasp_dpt, pred_decode_center_view_angle
        from models.grasp_designs import design_options, validate_checkpoint_design
        c = arguments.cfgs
        model = economicgrasp_dpt(
            **design_options(c), min_depth=c.min_depth, max_depth=c.max_depth,
            bin_num=c.bin_num, is_training=False, use_obs_depth=False,
            pose_depth_mode=str(c.pose_depth_mode),
            camera_pose_key=str(getattr(c, 'camera_pose_key', 'camera_pose_vec')),
            camera_gravity_key=str(getattr(c, 'camera_gravity_key', 'camera_gravity_vec')),
            pose_hidden_dim=int(getattr(c, 'pose_hidden_dim', 64)),
            ray_gravity_hidden_dim=int(getattr(c, 'ray_gravity_hidden_dim', 64)),
            ray_gravity_mid_dim=int(getattr(c, 'ray_gravity_mid_dim', 32)),
            use_cdf=True, vis_dir=None).to(device)
        # A locally supplied, hash-verified checkpoint is a trusted research artifact.
        data = torch.load(checkpoint, map_location='cpu', weights_only=False)
        validate_checkpoint_design(model, data)
        state = data['model_state_dict'] if isinstance(data, dict) and 'model_state_dict' in data else data
        model.load_state_dict(state, strict=True)
    settings = dict(max_width=float(namespace.grasp_max_width),
                    num_points=int(namespace.num_point), seed_mode=getattr(model, 'seed_selection_mode', None),
                    camera_pose_key=str(getattr(namespace, 'camera_pose_key', 'camera_pose_vec')),
                    camera_gravity_key=str(getattr(namespace, 'camera_gravity_key', 'camera_gravity_vec')),
                    model_args=config.get('model_args', []), factory='main',
                    design=getattr(model, 'grasp_design', None))
    return MainRuntime(model, actual_sha, settings, activation, pred_decode_center_view_angle)
