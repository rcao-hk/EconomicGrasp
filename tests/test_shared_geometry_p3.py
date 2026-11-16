"""P3 diagnostics CPU tests: no production CAD, test split, or server weights."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from shared_geometry.data import load_signed
from shared_geometry.geometry import DEFAULT_OFFSETS_M
from shared_geometry.p1_clean_visible import build
from shared_geometry.p2_geometry_diagnostics import fit_shared, compare_shared
from shared_geometry.p3_action_geometry_diagnostics import (
    SCHEMA, REGIONS, _flip_kind, _make_hybrid, _sanity_check_input,
    aggregate, decision_flips, diagnose, pair_geometry_evidence,
    stencil_regions,
)
from tests.test_shared_geometry_p0_p1 import synthetic_sources


class P3GeometryUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_regions_disjoint_and_complete(self):
        mask = stencil_regions()
        self.assertEqual({k:int(v.sum()) for k,v in mask.items()},
                         {'palm_proxy':15,'lateral_finger_proxy':24,'opening_core_proxy':36})
        np.testing.assert_array_equal(sum(mask.values()), np.ones(75))

    @staticmethod
    def geometry(udf=.2, state=1):
        x = np.zeros((7,75,6),dtype=np.float32)
        x[...,0] = udf
        x[...,1] = .1
        x[...,2+state] = 1
        return x

    def test_exact_action_relative_delta_mm_with_gripper_stencil_correspondence(self):
        clean = self.geometry()
        pred = clean.copy()
        palm = REGIONS['palm_proxy']
        pred[1,palm,0] += .1  # normalized 0.1 at +1 action -> 3 mm delta error
        got = pair_geometry_evidence(pred,clean,1,.03)
        self.assertAlmostEqual(got['palm_proxy']['udf_delta_mae_mm'],3.,places=5)
        self.assertAlmostEqual(got['all']['udf_delta_mae_mm'],.6,places=5)
        self.assertAlmostEqual(got['opening_core_proxy']['udf_delta_mae_mm'],0.)
        self.assertEqual(got['all']['n_both_observed'],75)
        self.assertEqual(got['all']['n_near_surface_both_observed'],75)
        self.assertAlmostEqual(got['all']['state_change_match_rate'],1.)
        # Same 3 mm absolute UDF shift at native and alternative should cancel
        # in delta: action-relative error is not absolute surface error.
        pred[0,palm,0] += .1
        corrected = pair_geometry_evidence(pred,clean,1,.03)
        self.assertAlmostEqual(corrected['all']['udf_delta_mae_mm'],0.,places=6)
        self.assertGreater(corrected['all']['native_udf_mae_mm'],0.)

    def test_unknown_is_not_mistaken_for_free_surface(self):
        x = self.geometry(state=3)
        got = pair_geometry_evidence(x,x,1,.03)
        self.assertEqual(got['all']['n_both_observed'],0)
        self.assertIsNone(got['all']['udf_delta_mae_mm'])
        self.assertIsNone(got['all']['near_udf_delta_mae_mm'])
        self.assertEqual(got['all']['n_clean_observed'],0)
        self.assertAlmostEqual(got['all']['state_pair_match_rate'],1.)
        self.assertAlmostEqual(got['all']['clean_visible_unobserved_fraction'],1.)

    def test_complete_region_swap_preserves_sixdim_observation_state(self):
        p = self.geometry(state=0)
        c = self.geometry(udf=.7,state=2)
        mixed = _make_hybrid(p,c,REGIONS['lateral_finger_proxy'])
        np.testing.assert_array_equal(mixed[:,REGIONS['lateral_finger_proxy']],c[:,REGIONS['lateral_finger_proxy']])
        np.testing.assert_array_equal(mixed[:,~REGIONS['lateral_finger_proxy']],p[:,~REGIONS['lateral_finger_proxy']])
        np.testing.assert_array_equal(mixed[...,2:].sum(-1),np.ones((7,75)))
        reverse = _make_hybrid(p,c,REGIONS['palm_proxy'],forward=False)
        np.testing.assert_array_equal(reverse[:,REGIONS['palm_proxy']],p[:,REGIONS['palm_proxy']])

    def test_flips_are_disjoint_and_respect_native_ties(self):
        q = np.asarray([6,4,4,4,4,4,4],dtype=np.int64)
        pred = np.asarray([1,2,0,0,0,0,0],dtype=float)
        clean = np.asarray([3,0,0,0,0,0,0],dtype=float)
        flip = decision_flips(pred,clean,q,DEFAULT_OFFSETS_M)
        self.assertTrue(flip['clean_better'])
        self.assertTrue(flip['flipped'])
        self.assertEqual(flip['flip_type'],'harm_to_keep')
        self.assertEqual(flip['pred_decision'],'harm')
        self.assertEqual(flip['clean_decision'],'keep')
        self.assertAlmostEqual(flip['source_gain_diff'],2/6)
        q = np.asarray([3,6,4,4,4,4,4],dtype=np.int64)
        pred = np.asarray([3,0,0,0,0,0,0],float)
        clean = np.asarray([1,3,0,0,0,0,0],float)
        self.assertEqual(decision_flips(pred,clean,q,DEFAULT_OFFSETS_M)['flip_type'],'keep_to_rescue')
        self.assertEqual(_flip_kind(1,1,q),'same_action')
        with self.assertRaises(ValueError):decision_flips(pred,clean,q[:-1],DEFAULT_OFFSETS_M)

    def test_conservative_joint_observation_and_invalid_shapes(self):
        p,c = self.geometry(),self.geometry()
        p[0,:10,2:]=0;p[0,:10,5]=1
        got=pair_geometry_evidence(p,c,1,.03)
        self.assertEqual(got['all']['n_both_observed'],65)
        self.assertEqual(got['all']['n_clean_observed'],75)
        with self.assertRaises(ValueError):pair_geometry_evidence(p[:2,:50],c[:2,:50],1,.03)
        with self.assertRaises(ValueError):_make_hybrid(p,c,np.zeros((74,),bool))


class P3Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_p3_synthetic_shared_weight_parity_full_smoke_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); tr,va,cfg=synthetic_sources(root)
            for part,idx in [('train',tr),('validation',va)]:
                build(SimpleNamespace(index=str(idx),targets=str(cfg),
                    output=str(root/'p1_cache'/part),resume=False,smoke=True,limit=0))
            fit_shared(SimpleNamespace(train_index=str(root/'p1_cache/train/index.json'),
                val_index=str(root/'p1_cache/validation/index.json'),
                output=str(root/'shared'),epochs=1,hidden=16,device='cpu',batch_size=2,
                val_batch_size=2,accumulate=1,lr=1e-3,weight_decay=1e-4,
                cpu_threads=1,seed=7,resume=False,smoke=True))
            checkpoint=str(root/'shared/best.pt')
            p2_path=str(root/'p2_compare.json')
            compare_shared(SimpleNamespace(index=str(root/'p1_cache/validation/index.json'),
                weight=checkpoint,device='cpu',batch_size=2,bootstrap=30,seed=0,
                smoke=True,output=p2_path))
            args=SimpleNamespace(index=str(root/'p1_cache/validation/index.json'),
                shared_weight=checkpoint,expected_p2=p2_path,device='cpu',
                batch_size=2,bootstrap=30,seed=0,skip_interventions=False,
                smoke=True,limit=0,output=str(root/'p3.json'))
            result=diagnose(args)
            self.assertEqual(result['schema'],SCHEMA)
            self.assertEqual(result['n_frames'],2)
            self.assertEqual(result['n_action_families'],4)
            self.assertTrue(result['source']['published_p2_gain_parity_checked'])
            self.assertTrue(result['privileged_clean_visible'])
            self.assertTrue(result['official_ap_not_run'])
            self.assertIn('by_offset_mm',result['summary'])
            self.assertIn('regions',result['summary'])
            self.assertEqual(sum(result['summary']['flip_type_counts'].values()),4)
            self.assertEqual(sum(x['pairs'] for x in result['summary']['by_offset_mm'].values()),4*6)
            self.assertEqual(len(result['per_family'][0]['alternatives']),6)
            self.assertLessEqual(result['region_interventions']['full_swap_logit_parity_max_error'],1e-5)
            self.assertEqual(set(result['region_interventions']['predicted_to_clean']),set(REGIONS))
            saved=load_signed(args.output)
            self.assertEqual(saved['n_action_families'],4)
            with self.assertRaises(FileExistsError):diagnose(args)
            args.output=str(root/'p3_no_swaps.json');args.skip_interventions=True
            no_swaps=diagnose(args)
            self.assertNotIn('region_interventions',no_swaps)
            self.assertAlmostEqual(no_swaps['summary']['pred_gain'],result['summary']['pred_gain'])
            # A genuine partial smoke must check checkpoint/index/teacher identity
            # but cannot be compared against the FULL P2 validation average.
            args.output=str(root/'partial_smoke.json');args.limit=1;args.smoke=True
            partial=diagnose(args)
            self.assertEqual(partial['n_frames'],1)
            self.assertFalse(partial['source']['published_p2_gain_parity_checked'])
            args.output=str(root/'forbidden.json');args.limit=1;args.smoke=False
            with self.assertRaises(ValueError):diagnose(args)
            args.smoke=True;args.limit=0
            args.expected_p2=str(root/'unrelated.json')
            with self.assertRaises(Exception):diagnose(args)

    def test_invalid_stencil_alignment_rejected(self):
        from shared_geometry.p1_clean_visible import make_cache_frame
        from tests.test_shared_geometry import fixture
        source=fixture(n=2)
        cached=make_cache_frame(source,source['pred_depth'],.03,.002)
        cached['local_xyz'][0,1,0,0]+=.003
        with self.assertRaises(ValueError):_sanity_check_input(cached,
            dict(partition='validation',scene=80,anno=0,domain='original'),
            dict(contract={'offsets_m':DEFAULT_OFFSETS_M}))


if __name__ == '__main__': unittest.main()