"""Synthetic CPU contract/regression tests. No checkpoint or CAD data required."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from diagnostics.gvar_fixed.contracts import (make_rows, exact_labels, physical_id,
    validate_grasps, write_csv, read_frames, VERSION)
from diagnostics.gvar_fixed.hooks import residual_permutation, reader_intervention
from diagnostics.gvar_fixed.runtime import MainRuntime
from diagnostics.gvar_fixed.analysis import (load_rows, analyze, scene_ci, pairwise_agreement,
    apply_calibration, comparison, family_statistics)
from diagnostics.gvar_fixed.calibration import fit_offsets
from diagnostics.gvar_fixed.run import dataset_indices


class ToySpatial(nn.Module):
    def forward(self, feat_2d, depth_map, K, image_hw, depth_prob=None, **kwargs):
        return feat_2d + depth_map * .03, {}


class ToySelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('view_dirs', torch.tensor([[0., 0., -1.], [0., 1., 0.]]))
    def _make_prob(self, score): return score.softmax(-1)
    def _select_view_indices(self, score, is_training, forced_view_inds=None):
        v = score.argmax(-1, keepdim=True)
        p = self._make_prob(score)
        return v, torch.zeros_like(v), p.gather(-1, v), 1, p
    @staticmethod
    def batch_viewpoint_params_to_matrix_fn(view, theta):
        r = torch.eye(3, device=theta.device).expand(len(theta), 3, 3).clone()
        r[:, 0, 0] = theta.cos(); r[:, 0, 1] = -theta.sin()
        r[:, 1, 0] = theta.sin(); r[:, 1, 1] = theta.cos()
        return r


class ToyGroup(nn.Module):
    def forward(self, seed_features, seed_xyz, depth_map, **kwargs):
        return seed_features + seed_xyz[..., 2][:, None] * .1 + depth_map.mean() * .01


class ToyReader(nn.Module):
    def __init__(self):
        super().__init__(); self.counterfactual_mode = 'r0'
    def set_counterfactual(self, value):
        if value not in ('r0', 'r1'): raise ValueError(value)
        self.counterfactual_mode = value
    def forward(self, queries, context):
        d = torch.arange(4, device=queries.device, dtype=queries.dtype).reshape(1, 1, 1, 4, 1)
        evidence = context['pregeom'].mean() + context['depth'].mean()
        delta = evidence * (d + 1) / 7
        if self.counterfactual_mode == 'r1': delta = evidence * torch.ones_like(d) / 7
        return queries + delta


class ToyCVA(nn.Module):
    def __init__(self, use_gvar):
        super().__init__()
        self.config = SimpleNamespace(mode='A1')
        self.selector = ToySelector(); self.group = ToyGroup()
        self.decoder = nn.Module(); self.decoder.reader = ToyReader() if use_gvar else None
        self.use_gvar = use_gvar
    def _call_view_net(self, seed_features, depth_map, end_points, **kwargs):
        score = seed_features.mean(1)
        end_points['view_score'] = torch.stack([score, -score], -1)
        return end_points, seed_features * .02 + depth_map.mean() * .01
    def forward(self, seed_features, seed_xyz, token_sel_idx, feat_map, depth_map, camera_K,
                end_points, is_training, **kwargs):
        ep, res = self._call_view_net(seed_features=seed_features, depth_map=depth_map,
                                     end_points=end_points, camera_K=camera_K, token_sel_idx=token_sel_idx)
        view, _, _, _, _ = self.selector._select_view_indices(ep['view_score'], is_training)
        ep['xyz_graspable'] = seed_xyz
        ep['token_sel_idx'] = token_sel_idx
        ep['grasp_top_view_inds'] = view[..., 0]
        ep['grasp_top_view_xyz'] = self.selector.view_dirs[view[..., 0]]
        b, c, q = seed_features.shape
        grouped = self.group(seed_features=seed_features + res, seed_xyz=seed_xyz,
                             depth_map=depth_map, camera_K=camera_K)
        action = grouped.mean(1).reshape(b, q, 1, 1, 1).expand(b, q, 2, 4, 3)
        action = action + torch.arange(4, device=action.device).reshape(1, 1, 1, 4, 1) * .05
        if self.decoder.reader is not None:
            action = self.decoder.reader(action, {'depth': depth_map, 'pregeom': ep['_gvar_pregeom']})
        score = action.mean(-1)
        ep['grasp_cdf_pred_angle_depth'] = score[:, None].expand(-1, 6, -1, -1, -1)
        ep['grasp_width_pred_angle_depth'] = (score.permute(0, 3, 1, 2) * .1 + .5)
        return ep


class ToyModel(nn.Module):
    def __init__(self, use_gvar, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.spatial_enhancer = ToySpatial(); self.kview_grasp_module = ToyCVA(use_gvar)
        self.enable_point_rayrope = self.use_cdf = True
        self.use_obs_depth = self.use_gt_xyz_for_train = False
        self.geometry_depth_source = 'pred'
    def forward(self, ep):
        image, K = ep['img'], ep['K']
        raw = image * self.scale
        depth = image[:, :1] * .01 * self.scale + .5
        enhanced, _ = self.spatial_enhancer(feat_2d=raw, depth_map=depth, K=K,
                                            image_hw=image.shape[-2:], depth_prob=None)
        idx = torch.tensor([[18, 27, 36, 45]], device=image.device)
        features = enhanced.reshape(1, 3, -1).gather(2, idx[:, None].expand(1, 3, 4))
        center = torch.zeros((1, 4, 3), device=image.device)
        center[..., 0] = (idx % 8).float() * .01
        center[..., 1] = (idx // 8).float() * .01
        center[..., 2] = depth.reshape(1, -1).gather(1, idx)
        if self.kview_grasp_module.decoder.reader is not None: ep['_gvar_pregeom'] = raw
        return self.kview_grasp_module(seed_features=features, seed_xyz=center, token_sel_idx=idx,
            feat_map=enhanced, depth_map=depth, camera_K=K, end_points=ep, is_training=False,
            process_grasp_labels_fn=None, process_grasp_labels_kwargs=None, topview_debug_fn=None,
            depth_prob=None, objectness_logits=None, graspness_map=None, img=image)


def runtime(gvar=True, scale=1):
    return MainRuntime(ToyModel(gvar, scale), ('a' if gvar else 'b') * 64,
                       dict(max_width=.1, num_points=100))


def image_batch():
    torch.manual_seed(5)
    return dict(img=torch.randn(1, 3, 8, 8), K=torch.eye(3)[None],
                gt_depth_m=torch.full((1, 1, 8, 8), 9.0))


def synthetic_rows(scenes=(100, 101), model_quality=None):
    out = []
    bank = np.zeros((2, 4, 17), dtype=np.float32)
    bank[..., 1] = .06; bank[..., 2] = .02; bank[..., 3] = np.arange(1, 5) * .01
    bank[..., 4:13] = np.eye(3).reshape(9); bank[..., 15] = .5; bank[..., 16] = -1
    bank[1, :, 13] = .1
    quality = np.array([[0., 1/6, 1., .5], [1/6, 0., .5, 1.]])
    bits = np.array([[[int(i >= 6-int(round(q*6))) for i in range(6)] for q in fam] for fam in quality])
    labels = dict(quality=quality, success=bits, empty=quality == 0,
                  collision=np.zeros((2, 4), bool), friction=np.where(quality > 0, (7 - np.rint(quality * 6)) * .2, -1))
    scores = dict(point=np.array([[.9, .8, .7, .6], [.8, .9, .7, .6]]),
                  gvar=np.array([[.1, .3, .9, .5], [.2, .1, .4, .9]]))
    for scene in scenes:
        frame = dict(sample=f'scene{scene:04d}_frame0000', scene=scene, anno=0, split='seen')
        for ref in ('point', 'gvar'):
            for model in ('point', 'gvar'):
                conditions = ('native', 'rgb_fixed', 'perm_cycle') if model == 'gvar' else ('native',)
                for cond in conditions:
                    s = scores[model] if cond == 'native' else scores['point']
                    out.extend(make_rows(frame, ref, bank, [0, 1], [0, 0], labels, s, model,
                                         cond, 'geom', ('a' if model == 'gvar' else 'b') * 64,
                                         ('a' if ref == 'gvar' else 'b') * 64))
    return out


class ContractsTests(unittest.TestCase):
    def test_physical_id_excludes_score_but_not_width(self):
        a = np.zeros(17, np.float32); a[1:4] = [.06, .02, .01]
        a[4:13] = np.eye(3).flatten(); a[15] = .5
        b = a.copy(); b[0] = .99
        self.assertEqual(physical_id(a), physical_id(b))
        b[1] += .001; self.assertNotEqual(physical_id(a), physical_id(b))

    def test_evaluator_deduplicates_and_retains_failures(self):
        rows = synthetic_rows((100,))
        bank = np.array([[0, *[float(r[k]) for k in ['width_m','height_m','insertion_depth_m'] +
                      [f'r{i}{j}' for i in range(3) for j in range(3)] + ['center_x','center_y','center_z']], -1]
                      for r in rows[:4]], np.float32)[None]
        class Eval:
            def evaluate(self, scene, anno, actions):
                self.n = len(actions)
                return SimpleNamespace(friction=np.array([.2,-1,.4,-1]), empty=np.array([0,1,0,0]),
                    pure_collision=np.array([0,0,0,1]), collision_or_empty=np.array([0,1,0,1]), stats={})
        e = Eval(); labels = exact_labels(e, 100, 0, np.repeat(bank, 2, 0))
        self.assertEqual(e.n, 4); self.assertEqual(labels['quality'].shape, (2,4))
        self.assertEqual(int(labels['empty'].sum()), 2)

    def test_frames_and_no_implicit_subset_order(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'frames.csv'; p.write_text('scene_id,anno_id,split\n100,0,seen\n')
            self.assertEqual(read_frames(p)[0]['scene'],100)
            p.write_text('scene_id,anno_id,split\n100,0,novel\n')
            with self.assertRaises(ValueError): read_frames(p)
        class Data:
            def __len__(self): return 256
            def scene_list(self): return ['scene0100']*256
        self.assertEqual(dataset_indices(Data())[(100,128)],128)


class HooksTests(unittest.TestCase):
    def test_only_residual_swapped_and_identity_bitwise(self):
        q = torch.arange(24.).reshape(1,1,2,4,3)
        delta = torch.arange(4.).reshape(1,1,1,4,1)
        y = q + delta
        self.assertIs(residual_permutation(q,y,[0,1,2,3]),y)
        expected = q + delta.index_select(3,torch.tensor([1,2,3,0]))
        torch.testing.assert_close(residual_permutation(q,y,[1,2,3,0]),expected)
        with self.assertRaises(ValueError): residual_permutation(q,y,[1,1,2,3])

    def test_restore_after_exception(self):
        reader = ToyReader().eval()
        with torch.inference_mode():
            with self.assertRaises(RuntimeError):
                with reader_intervention(reader,'rgb_fixed'):
                    self.assertEqual(reader.counterfactual_mode,'r1'); raise RuntimeError('test')
        self.assertEqual(reader.counterfactual_mode,'r0')
        self.assertEqual(len(reader._forward_hooks),0)
        with self.assertRaises(RuntimeError):
            with reader_intervention(reader,'native'): pass


class ReplayTests(unittest.TestCase):
    def test_native_parity_and_no_state_or_gradient_change(self):
        r=runtime(); state={k:v.clone() for k,v in r.model.state_dict().items()}
        snap=r.capture(image_batch()); audit=r.validate_native(snap)
        self.assertEqual(audit['native_max_abs_errors']['grasp_cdf_pred_angle_depth'],0)
        self.assertNotIn('gt_depth_m',snap.input_hashes)
        for k,v in state.items(): torch.testing.assert_close(v,r.model.state_dict()[k],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in r.model.parameters()))
        self.assertNotIn('_select_view_indices', r.cva.selector.__dict__)
        self.assertNotIn('_call_view_net', r.cva.__dict__)

    def test_cross_reference_recomputes_own_features_with_shared_geometry(self):
        p,g=runtime(False,2),runtime(True,1)
        batch=image_batch(); ps,gs=p.capture(batch),g.capture(batch)
        self.assertNotEqual(ps.geometry_hash,gs.geometry_hash)
        p.validate_native(ps); g.validate_native(gs)
        out,audit=g.replay(gs,ps)
        self.assertEqual(audit['common_geometry_sha256'],ps.geometry_hash)
        self.assertEqual(audit['routes'],dict(spatial=1,view=1,group=1,reader=1))
        bank,q,angle=p.make_bank(ps,2,10)
        scores,widths=g.selected_outputs(out,q,angle)
        self.assertEqual(scores.shape,(2,4)); self.assertEqual(widths.shape,(2,4))
        validate_grasps(bank)
        altered=copy.copy(gs); altered.image_hash='wrong'
        with self.assertRaises(ValueError): g.replay(altered,ps)

    def test_interventions_change_scores_not_bank_and_cleanup(self):
        r=runtime(); snap=r.capture(image_batch())
        base,_=r.replay(snap,snap)
        altered,_=r.replay(snap,snap,'rgb_fixed')
        perm,_=r.replay(snap,snap,'perm_cycle')
        bias,_=r.replay(snap,snap,'geometry_bias_5')
        self.assertFalse(torch.equal(base['grasp_cdf_pred_angle_depth'],altered['grasp_cdf_pred_angle_depth']))
        self.assertFalse(torch.equal(base['grasp_cdf_pred_angle_depth'],perm['grasp_cdf_pred_angle_depth']))
        self.assertFalse(torch.equal(base['grasp_cdf_pred_angle_depth'],bias['grasp_cdf_pred_angle_depth']))
        again,_=r.replay(snap,snap)
        torch.testing.assert_close(base['grasp_cdf_pred_angle_depth'],again['grasp_cdf_pred_angle_depth'],atol=0,rtol=0)
        with self.assertRaises(ValueError): r.replay(snap,snap,'unknown')
        self.assertEqual(r.reader.counterfactual_mode,'r0')
        self.assertEqual(len(r.reader._forward_hooks),0)
        self.assertEqual(len(r.reader._forward_pre_hooks),0)


class AnalysisTests(unittest.TestCase):
    def load(self,rows):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'scores.csv';write_csv(p,rows); return load_rows([p])

    def test_paired_selection_gain_regret_identity_and_coverage(self):
        data=analyze(self.load(synthetic_rows()),100,5)
        c=data['contrasts']['point/gvar_native-minus-point_native']['metrics']
        self.assertAlmostEqual(c['selected_quality_gain']['mean'],1.0)
        self.assertAlmostEqual(c['regret_reduction']['mean'],1.0)
        self.assertEqual(c['repair']['mean'],1)
        self.assertEqual(c['selected_quality_gain']['scenes'],2)
        self.assertEqual(pairwise_agreement([1,2,3,4],[0,0,0,0]),(None,0))

    def test_refuse_label_geometry_checkpoint_or_bank_changes(self):
        for key,value in [('width_m',.07),('quality',.9),('geometry_sha256','other'),('bank_id','other')]:
            rows=synthetic_rows();rows[8][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.load(rows)

    def test_reject_duplicate_or_incomplete_families(self):
        rows=synthetic_rows()
        with self.assertRaises(ValueError):self.load(rows+rows[:1])
        with self.assertRaises(ValueError):self.load(rows[1:])
        rows=[r for r in rows if not(r['model']=='point' and r['sample_id']=='scene0100_frame0000')]
        with self.assertRaises(ValueError): analyze(self.load(rows),10)

    def test_equal_frame_scene_weighting_and_single_scene_ci(self):
        r=[dict(scene=100,split='seen',sample='a',v=0)]*100
        r += [dict(scene=100,split='seen',sample='b',v=1)]
        r += [dict(scene=101,split='seen',sample='c',v=1)]
        stat=scene_ci(r,'v',100,3)
        self.assertAlmostEqual(stat['mean'],.75)
        self.assertIsNone(scene_ci(r[:100],'v',100)['ci95'])

    def test_calibration_requires_disjoint_scenes_and_checkpoint(self):
        val=self.load(synthetic_rows((102,103)))
        fitted=fit_offsets(val)
        test=self.load(synthetic_rows())
        calibrated=apply_calibration(test,fitted)
        self.assertGreater(len(calibrated),len(test))
        data=analyze(calibrated,20)
        self.assertIn('point/calibrated_gvar-minus-calibrated_point',data['contrasts'])
        with self.assertRaises(ValueError):apply_calibration(val,fitted)
        wrong=copy.deepcopy(fitted); wrong['parameters']['point|point']['checkpoint_sha256']='wrong'
        with self.assertRaises(ValueError):apply_calibration(test,wrong)

    def test_refuse_relabelled_success_and_scene_mix(self):
        for key,value in [('friction',.2),('scene_id',105),('split','novel')]:
            rows=synthetic_rows();rows[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.load(rows)

    def test_geometry_bias_does_not_mutate_input_context(self):
        reader=ToyReader().eval();depth=torch.ones(1,1,2,2)
        context={'depth':depth,'pregeom':torch.zeros(1,3,2,2)}
        q=torch.zeros(1,1,1,4,2)
        with torch.inference_mode():
            native=reader(q,context)
            with reader_intervention(reader,'geometry_bias_5'):
                changed=reader(q,context)
        torch.testing.assert_close(depth,torch.ones_like(depth),atol=0,rtol=0)
        self.assertFalse(torch.equal(native,changed))
        self.assertEqual(len(reader._forward_pre_hooks),0)

    def test_complete_cli_requires_expected_frames(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);write_csv(p/'input.csv',synthetic_rows())
            proc=subprocess.run([sys.executable,'-m','diagnostics.gvar_fixed.analysis',
                '--csv',str(p/'input.csv'),'--output',str(p/'bad.json'),
                '--expected-frames','18','--require-cross-model','--bootstrap','5'],
                capture_output=True,text=True)
            self.assertNotEqual(proc.returncode,0)
            self.assertFalse((p/'bad.json').exists())

    def test_cli_end_to_end_json_markdown(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); write_csv(p/'input.csv',synthetic_rows())
            result=subprocess.run([sys.executable,'-m','diagnostics.gvar_fixed.analysis',
                '--csv',str(p/'input.csv'),'--output',str(p/'result.json'),
                '--markdown',str(p/'report.md'),'--bootstrap','50'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            payload=json.loads((p/'result.json').read_text())
            self.assertEqual(len(payload['treatments']),8)
            self.assertIn('不是官方 AP',(p/'report.md').read_text())


if __name__=='__main__': unittest.main()
