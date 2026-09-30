"""Live frozen MGF source and scoring controls. No feature/action-label cache I/O.

The source always runs in eval/no_grad. Small per-forward dictionaries live only
until their consumer completes. Depth and ray-profile predictions remain frozen.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from mgf_p0_core import VERSION, VARIANTS, freeze, assert_frozen, fingerprint


def code_digest():
    from metric_field_runtime import code_fingerprint
    root = Path(__file__).resolve().parent
    h = hashlib.sha256(code_fingerprint().encode())
    for p in sorted(root.glob('mgf_p0_*.py')):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


class FrozenSource(nn.Module):
    def __init__(self, checkpoint, device='cuda'):
        super().__init__()
        from metric_field_runtime import construct_model, sha256_file, VERSION as MGF_VERSION
        self.checkpoint = str(Path(checkpoint).resolve())
        self.checkpoint_sha = sha256_file(checkpoint)
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if ck.get('version') != MGF_VERSION or ck['protocol'].get('partial_run', True):
            raise ValueError('P0 requires a complete trusted MGF source checkpoint')
        self.protocol = ck['protocol']
        self.epoch = int(ck['epoch'])
        official = Path('checkpoints') / f"depth_anything_v2_{self.protocol['encoder']}.pth"
        if sha256_file(official) != self.protocol['dav2_sha256']:
            raise RuntimeError('Official DAV2 weights differ from source training')
        if self.protocol['seed_mode'] != 'image_fps':
            raise ValueError('P0 fixed-action protocol requires image_fps')
        self.model = construct_model(self.protocol, device=device, top4=False)
        self.model.load_state_dict(ck['model'], strict=True)
        del ck
        freeze(self.model)
        self.config = self.model.config
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        if hasattr(self, 'model'):
            freeze(self.model)
        return self

    @torch.no_grad()
    def forward(self, batch, fixed=None, geometry_from=None):
        """Freeze candidate bank when fixed is given; replace only RGB-predicted
        depth/profile when geometry_from is given. Never use external depth.
        Labels (P0-1 only) are attached after prediction by main's online matcher.
        """
        from models.economicgrasp_metric_field import (
            filter_empty_grasp_objects, assert_object_payloads_cpu, build_candidate_actions)
        assert_frozen(self.model)
        model = self.model
        model._depth_pack = model._proposal_feature = None
        ep = dict(batch)
        if 'object_poses_list' in ep:
            if fixed is not None or geometry_from is not None:
                raise ValueError('Counterfactuals must use exact labels, not native annotation transfer')
            ep, _ = filter_empty_grasp_objects(ep)
            assert_object_payloads_cpu(ep)
        ep['cva_force_process_grasp_labels'] = 'object_poses_list' in ep
        ep['cva_compute_diagnostics'] = False
        captured, handles = {}, []
        local = model.base.kview_grasp_module

        def grouped(_m, args):
            captured['grouped'] = args[0].detach()
        handles.append(local.decoder.register_forward_pre_hook(grouped))

        if fixed is not None:
            def fix_seeds(_m, args, kw):
                if args:
                    raise RuntimeError('P0 expects keyword-based CVA forward API')
                kw = dict(kw)
                idx = fixed['tokens'].to(kw['feat_map'].device)
                fmap = kw['feat_map'].flatten(2)
                kw['seed_features'] = fmap.gather(2, idx[:,None].expand(-1,fmap.shape[1],-1))
                kw['seed_xyz'] = fixed['centres']
                kw['token_sel_idx'] = idx
                kw['is_training'] = False
                e = kw['end_points']
                e['xyz_graspable'] = e['token_sel_xyz'] = fixed['centres']
                e['token_sel_idx'] = idx
                return args, kw
            handles.append(local.register_forward_pre_hook(fix_seeds, with_kwargs=True))

            def fix_views(_m, args, kw):
                if args:
                    raise RuntimeError('P0 expects keyword-based selector API')
                kw = dict(kw)
                score = _m._normalize_view_score_shape(kw['view_score'])
                ids = fixed['views'].to(score.device)
                # forced_view_inds is ignored by main's eval/top1 strategy.
                # Force argmax explicitly rather than switching source to train.
                kw['view_score'] = F.one_hot(ids, num_classes=score.shape[-1]).to(score)
                kw['forced_view_inds'] = None
                kw['is_training'] = False
                return args, kw
            handles.append(local.selector.register_forward_pre_hook(fix_views, with_kwargs=True))

        if geometry_from is not None:
            def depth_swap(_m, args, output):
                out = list(output)
                out[0] = geometry_from['depth']
                out[1] = F.interpolate(out[0], size=output[1].shape[-2:], mode='nearest')
                # Leave latent metric features and encoder features with the
                # current visual observation. Only numeric geometry is swapped.
                pack = list(model._depth_pack)
                pack[0], pack[1] = out[0], out[1]
                model._depth_pack = tuple(pack)
                return tuple(out)
            handles.append(model.base.depth_net.register_forward_hook(depth_swap))
        try:
            ep = model.base(ep)
            if 'grouped' not in captured or model._depth_pack is None or model._proposal_feature is None:
                raise RuntimeError('Frozen source interface changed; required live features missing')
            depth, _, metric, _, feats, _ = model._depth_pack
            h, w = batch['img'].shape[-2:]
            relative, relative_raw = model.relative_decoder(feats, h//14, w//14)
            _, prob = model.ray_head(relative, F.relu(relative_raw), metric, depth, (h,w), batch['K'])
            if geometry_from is not None:
                prob = geometry_from['prob']
            actions, shape = build_candidate_actions(ep, model.rotation_fn, model.max_width)
            if fixed is not None:
                torch.testing.assert_close(ep['xyz_graspable'], fixed['centres'], atol=0, rtol=0)
                torch.testing.assert_close(ep['token_sel_idx'], fixed['tokens'], atol=0, rtol=0)
                torch.testing.assert_close(ep['grasp_top_view_inds'], fixed['views'], atol=0, rtol=0)
            return {'ep': ep, 'grouped': captured['grouped'],
                    'proposal': model._proposal_feature.detach(), 'relative': relative.detach(),
                    'metric': metric.detach(), 'depth': depth.detach(), 'prob': prob.detach(),
                    'K': batch['K'].detach(), 'hw': (h,w), 'actions': actions.detach(), 'shape': shape}
        finally:
            for h in handles: h.remove()
            model._depth_pack = model._proposal_feature = None


def fixed_queries(context, indices):
    e = context['ep']
    return {'tokens': e['token_sel_idx'][:,indices].detach().clone(),
            'centres': e['xyz_graspable'][:,indices].detach().clone(),
            'views': e['grasp_top_view_inds'][:,indices].detach().clone()}


def make_control(source, variant):
    """Identical latent/support architecture and initialization for full/feature.
    CVA is an independently copied complete scoring decoder. Its width outputs
    are NEVER used; every condition executes source-predicted physical widths.
    """
    from metric_grasp_field_core import GraspFieldReadout
    from metric_grasp_field_core import gripper_support, project_points, sample_map, evidence_at

    class EvidenceReadout(GraspFieldReadout):
        def __init__(self, cfg, evidence):
            super().__init__(cfg)
            self.evidence = bool(evidence)

        def _chunk(self, feature, prob, actions, K, image_hw):
            xyz, local, roles = gripper_support(actions.float())
            uv, valid = project_points(xyz, K.float(), image_hw)
            valid = valid & (xyz[...,2] >= self.cfg.min_depth) & (xyz[...,2] <= self.cfg.max_depth)
            visual = sample_map(feature.float(), uv, image_hw)
            if self.evidence:
                p = sample_map(prob.detach().float(), uv, image_hw)
                evidence = evidence_at(p, xyz[...,2], self.cfg)
            else:
                # The four explicit ray channels alone are ablated. Shapes,
                # support positions, role IDs, visibility mask and latents match.
                evidence = visual.new_zeros(*visual.shape[:-1], 4)
            evidence = torch.cat((evidence, valid[...,None].float()), -1)
            tokens = self.point(torch.cat((visual, evidence, local/.1), -1))
            tokens = (tokens + self.role_embed(roles)[None,None]) * valid[...,None]
            pooled = []
            for role in range(5):
                m = (roles == role)[None,None] & valid
                pooled.append((tokens*m[...,None]).sum(-2)/m.sum(-1,keepdim=True).clamp_min(1))
            size = actions[...,1:4]/actions.new_tensor([.1,.02,.04])
            return self.head(torch.cat((*pooled,size), -1))

    class Control(nn.Module):
        def __init__(self):
            super().__init__()
            if variant not in VARIANTS: raise ValueError(variant)
            self.variant = variant
            if variant in ('feature_only', 'full'):
                self.adapter = copy.deepcopy(source.model.task_adapter).requires_grad_(True)
                self.reader = EvidenceReadout(source.config, variant == 'full')
                self.reader.load_state_dict(source.model.readout.state_dict(), strict=True)
                self.reader.zero_initialize_output()
                self.reader.requires_grad_(True)
            elif variant == 'cva':
                self.decoder = copy.deepcopy(source.model.base.kview_grasp_module.decoder)
                self.decoder.requires_grad_(True)
                # Other decoder components, including width feature tokens used
                # by score attention, belong to this PRIVATE scoring copy.
                self.decoder.width_head.requires_grad_(False)

        def forward(self, context, actions=None, base_logits=None):
            base = context['ep']['grasp_cdf_pred_angle_depth'] if base_logits is None else base_logits
            if self.variant == 'base': return base.detach()
            b, q, a, d = context['shape']
            if self.variant == 'cva':
                e = {'kview_angle_query_base_q': q, 'kview_angle_query_num_angle': a}
                return self.decoder(context['grouped'].detach(), e)['grasp_cdf_pred_angle_depth']
            from metric_grasp_field_core import compose_monotone_residual_logits
            feature = self.adapter(context['proposal'], context['relative'], context['metric'],
                                   context['prob'].shape[-2:])
            act = context['actions'] if actions is None else actions
            raw = self.reader(feature, context['prob'], act, context['K'], context['hw'], raw_output=True)
            raw = raw.reshape(b,q,a,d,6)
            return compose_monotone_residual_logits(base.detach().movedim(1,-1),raw).movedim(-1,1).contiguous()

    return Control().to(next(source.parameters()).device)


def load_control(source, path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('version') != VERSION or ck['protocol']['source_sha256'] != source.checkpoint_sha:
        raise RuntimeError('Control checkpoint/source mismatch')
    variant = ck['protocol']['variant']
    control = make_control(source,variant)
    control.load_state_dict(ck['control'],strict=True)
    control.eval()
    return control, ck


def safe_batch(raw, device, labels):
    from metric_field_runtime import move_batch
    result = move_batch(raw, device, inference=not labels)
    # Do not trust optional extra numeric depth keys to select model geometry.
    for k in ('depth', 'sensor_depth_m', 'obs_depth_m', 'input_depth'):
        result.pop(k, None)
    return result


def paired_dataset(original, rgb_root):
    """RGB replacement ONLY: preserve original crop, depth paths and intrinsics.
    GN-Trans layout is the existing scenes/SSSSS/FFFF_color.png convention.
    Never instantiate its depth-based crop as the observation control.
    """
    trans = copy.copy(original)
    trans.colorpath = [str(Path(rgb_root)/'scenes'/f'{int(str(s).split("_")[-1]):05d}'/
                           f'{int(a):04d}_color.png')
                       for s,a in zip(original.scenename, original.frameid)]
    return trans


def verify_pair(original, material, index, raw_o, raw_m):
    from PIL import Image
    p, q = original.colorpath[index], material.colorpath[index]
    if not Path(q).is_file(): raise FileNotFoundError(q)
    with Image.open(p) as a, Image.open(q) as b:
        if a.size != b.size: raise RuntimeError(f'Paired RGB dimensions differ: {p}={a.size}, {q}={b.size}')
    for key in ('K', 'img'):
        if raw_o[key].shape != raw_m[key].shape: raise RuntimeError(f'Paired {key} shapes differ')
    torch.testing.assert_close(torch.as_tensor(raw_o['K']), torch.as_tensor(raw_m['K']), atol=1e-6, rtol=0)
    # Matching metadata cannot prove rendering registration; that is a dataset
    # contract. It does prevent re-cropping with virtual/sensor depth differences.
