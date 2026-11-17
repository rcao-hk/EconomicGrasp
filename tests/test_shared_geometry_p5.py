"""P5 offline diagnostics and reader adaptation, using only synthetic CPU data."""
from __future__ import annotations
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as A
import numpy as np
import torch

from shared_geometry.data import load_signed, atomic_json, signed
from shared_geometry.p4_depth_data import read_depth_index
from shared_geometry.p5_geometry_diagnostics import (
    depth_edges,project_support,pixel_split,nearest_pixel_drift,
    geometry_regional,family_geometry_features,run as diagnose)
from shared_geometry.p5_reader_adaptation import (
    build,read_index,read_sample,collate,fit,compare,forward,load_reader)
from tests.test_shared_geometry_p4 import setup_sources,train_args
from shared_geometry.p4_depth_baseline import fit as fit_corrector,evaluate_actions
from types import SimpleNamespace


class P5Units(unittest.TestCase):
    def test_edge_detection_and_missing_depth(self):
        x=np.ones((6,6),np.float32)*.4
        x[:,3:]=.5;x[0,0]=0
        edge=depth_edges(x,.02)
        self.assertTrue(edge[3,2]);self.assertTrue(edge[3,3]);self.assertTrue(edge[0,1]);self.assertFalse(edge[5,0])

    def test_pixel_split_support_edge_and_smooth_are_disjoint(self):
        shape=(10,10);clean=np.ones(shape,np.float32)*.5
        pred=clean.copy();pred[:,:5]+=.02
        corr=clean.copy();corr[:,:5]+=.01
        K=np.asarray([[10,0,5],[0,10,5],[0,0,1]],np.float32)
        xyz=np.array([[0,0,.5],[.1,0,.5]],np.float32)
        r=pixel_split(pred,corr,clean,xyz,K,radius=0)
        self.assertEqual(r['full']['n'],100)
        self.assertAlmostEqual(r['full']['improvement_mm'],5.,places=5)
        self.assertEqual(r['projected_support']['n']+r['away_from_support']['n'],100)
        self.assertEqual(r['support_and_edge']['n']+r['support_and_smooth']['n'],r['projected_support']['n'])

    def test_geometry_delta_bias_cancels_but_absolute_does_not(self):
        clean=np.zeros((1,7,75,6),np.float32);clean[...,2]=1
        pred=clean.copy();corrected=clean.copy()
        pred[...,0]=.1;corrected[...,0]=.05
        r=geometry_regional(pred,corrected,clean,.03)
        self.assertAlmostEqual(r['predicted']['all']['udf_mae_mm'],3.,places=5)
        self.assertAlmostEqual(r['corrected']['all']['udf_mae_mm'],1.5,places=5)
        self.assertAlmostEqual(r['predicted']['all']['udf_relative_mae_mm'],0.)
        f=family_geometry_features(pred,corrected,clean,.03)
        self.assertAlmostEqual(f[0]['relative_udf_error_improvement_mm'],0.)

    def test_empty_mask_not_fake_zero(self):
        clean=np.zeros((1,7,75,6),np.float32);clean[...,5]=1
        pred=clean.copy();corr=clean.copy()
        r=geometry_regional(pred,corr,clean,.03)
        self.assertIsNone(r['corrected']['all']['udf_mae_mm'])
        self.assertIsNone(family_geometry_features(pred,corr,clean,.03)[0]['relative_udf_error_improvement_mm'])

    def test_nearest_pixel_drift_identity(self):
        dep=np.ones((8,8),np.float32)*.5
        K=np.asarray([[15,0,4],[0,15,4],[0,0,1]],np.float32)
        pts=np.array([[0,0,.5],[.01,.01,.5]],np.float32)
        r=nearest_pixel_drift(dep,dep,pts,K)
        self.assertEqual(r['changed_pixel_fraction'],0.)
        self.assertEqual(r['mean_nearest_pixel_shift'],0.)


class P5SyntheticE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.tr,cls.va,cls.cfg=setup_sources(cls.root)
        fit_corrector(train_args(cls.root,'correction',epochs=2))
        e=A(depth_index=str(cls.root/'depth/validation/index.json'),
            p1_index=str(cls.root/'p1/validation/index.json'),original_index=str(cls.va),
            weight=str(cls.root/'correction/best.pt'),
            shared_weight=str(cls.root/'reader/best.pt'),expected_p2=str(cls.root/'p2.json'),
            output=str(cls.root/'p4eval'),device='cpu',batch_size=2,bootstrap=20,seed=0,limit=0,smoke=True)
        evaluate_actions(e)
        for part in ('train','validation'):
            build(A(partition=part,depth_index=str(cls.root/f'depth/{part}/index.json'),
                p1_index=str(cls.root/f'p1/{part}/index.json'),depth_weight=str(cls.root/'correction/best.pt'),
                output=str(cls.root/f'p5_cache/{part}'),resume=False,limit=0,smoke=True,device='cpu'))

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def train_spec(self,arm,out,epochs=2):
        return A(arm=arm,train_index=str(self.root/'p5_cache/train/index.json'),
            val_index=str(self.root/'p5_cache/validation/index.json'),
            output=str(self.root/out),device='cpu',epochs=epochs,batch_size=2,accumulate=1,
            val_batch_size=2,hidden=8,seed=11,lr=1e-3,weight_decay=1e-4,cpu_threads=1,
            resume=False,smoke=True)

    def test_cache_only_stores_corrected_geometry_and_detects_tampering(self):
        r=read_index(self.root/'p5_cache/validation/index.json','validation',True)
        self.assertEqual(len(r['entries']),2)
        e=r['entries'][0]
        with np.load(e['corrected_path'],allow_pickle=False) as z:
            self.assertNotIn('visible_depth',z.files)
            self.assertNotIn('feature_map',z.files)
            self.assertNotIn('pred_depth',z.files)
            self.assertEqual(z['corrected_physics'].shape,(2,7,75,6))
        f=read_sample(e)
        self.assertTrue(np.array_equal(f['quality_count'],f['quality_count']))
        idx=copy.deepcopy(r);idx.pop('fingerprint');idx['entries'][0]['corrected_sha256']='bad'
        path=self.root/'invalid_p5_index.json';atomic_json(path,signed(idx))
        with self.assertRaises(ValueError):read_index(path,'validation',True)
        with self.assertRaises(ValueError):read_index(self.root/'p5_cache/validation/index.json','validation',False)

    def test_batch4_reader_gradient_equals_four_independent_images(self):
        from shared_geometry.model import PredictedGeometryControl,ModelConfig,quality_loss
        torch.manual_seed(10)
        reader=PredictedGeometryControl(ModelConfig(feature_channels=1,hidden=8,truncation_m=.03))
        idx=read_index(self.root/'p5_cache/validation/index.json','validation',True)
        one=read_sample(idx['entries'][0]);two=read_sample(idx['entries'][1])
        samples=[one,two,one,two]
        batch=collate(samples)
        z=forward(reader,batch,'corrected','cpu')
        solo=torch.stack([forward(reader,collate([s]),'corrected','cpu')[0] for s in samples])
        torch.testing.assert_close(z,solo,atol=1e-6,rtol=0)
        q=torch.from_numpy(batch['quality_count']).long()
        loss,_=quality_loss(z,q)
        gp=torch.autograd.grad(loss,tuple(reader.parameters()),retain_graph=True)
        l2=sum(quality_loss(solo[i],q[i])[0] for i in range(4))/4
        gs=torch.autograd.grad(l2,tuple(reader.parameters()))
        for a,b in zip(gp,gs):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)

    def test_interrupted_reader_train_resume_exact_weights(self):
        a=self.train_spec('corrected','reader_full_resume',epochs=2)
        fit(a)
        b=self.train_spec('corrected','reader_partial_resume',epochs=2)
        b.stop_after_epoch=1;fit(b)
        partial=load_signed(self.root/'reader_partial_resume/selection.json')
        self.assertFalse(partial['complete'])
        b.stop_after_epoch=0;b.resume=True;fit(b)
        full=torch.load(self.root/'reader_full_resume/last.pt',map_location='cpu',weights_only=False)
        rest=torch.load(self.root/'reader_partial_resume/last.pt',map_location='cpu',weights_only=False)
        self.assertEqual(full['updates'],rest['updates'])
        for key in full['model']:
            torch.testing.assert_close(full['model'][key],rest['model'][key],rtol=0,atol=0)
        self.assertTrue(load_signed(self.root/'reader_partial_resume/selection.json')['complete'])

    def test_forward_batch_is_image_independent(self):
        from shared_geometry.model import PredictedGeometryControl,ModelConfig
        torch.manual_seed(8);m=PredictedGeometryControl(ModelConfig(feature_channels=1,hidden=8,truncation_m=.03))
        idx=read_index(self.root/'p5_cache/validation/index.json','validation',True)
        samples=[read_sample(e) for e in idx['entries']]
        batch=collate(samples)
        z=forward(m,batch,'corrected','cpu')
        torch.testing.assert_close(z,torch.stack([forward(m,collate([a]),'corrected','cpu')[0] for a in samples]),atol=1e-6,rtol=0)
        with self.assertRaises(ValueError):forward(m,batch,'visible_depth','cpu')

    def test_matched_training_resume_and_comparison(self):
        for arm in ('predicted','corrected'):
            a=self.train_spec(arm,'fitted_'+arm)
            fit(a)
            ck=torch.load(self.root/('fitted_'+arm)/'last.pt',weights_only=False)
            self.assertEqual(ck['updates'],2)
            self.assertEqual(ck['protocol']['parameters'],sum(p.numel() for p in __import__('shared_geometry.model',fromlist=['PredictedGeometryControl']).PredictedGeometryControl(__import__('shared_geometry.model',fromlist=['ModelConfig']).ModelConfig(feature_channels=1,hidden=8,truncation_m=.03)).parameters()))
        a=self.train_spec('corrected','fitted_corrected');a.resume=True;fit(a)
        a.lr=.4
        with self.assertRaises(ValueError):fit(a)
        out=self.root/'comparison'
        c=A(index=str(self.root/'p5_cache/validation/index.json'),
            p1_index=str(self.root/'p1/validation/index.json'),
            predicted_weight=str(self.root/'fitted_predicted/best.pt'),
            corrected_weight=str(self.root/'fitted_corrected/best.pt'),
            shared_weight=str(self.root/'reader/best.pt'),expected_p4=str(self.root/'p4eval/summary.json'),
            output=str(out),device='cpu',batch_size=2,bootstrap=20,seed=0,smoke=True)
        result=compare(c)
        self.assertEqual(set(result['scores']),{
            'predicted_p2_fixed','corrected_p2_fixed','clean_p2_fixed','predicted_retrained',
            'corrected_adapted','pred_reader_on_corrected','corrected_reader_on_predicted'})
        self.assertTrue(result['cross_source_scores_are_OOD'])
        self.assertEqual(len(result['paired']),6)
        self.assertIn('difference_in_differences_geometry_adaptation',result['paired'])
        with self.assertRaises(FileExistsError):compare(c)

    def test_p50_reuses_exact_p4_depth_and_quality_parity(self):
        a=A(depth_index=str(self.root/'depth/validation/index.json'),
            p1_index=str(self.root/'p1/validation/index.json'),original_index=str(self.va),
            p4_summary=str(self.root/'p4eval/summary.json'),shared_weight=str(self.root/'reader/best.pt'),
            output=str(self.root/'p5_geometry'),device='cpu',batch_size=2,bootstrap=20,
            edge_threshold_m=.02,support_radius_px=2,limit=0,smoke=True,seed=0)
        result=diagnose(a)
        self.assertEqual(result['n_frames'],2)
        self.assertEqual(result['n_families'],4)
        self.assertTrue(result['claims']['privileged_clean_visible_offline_only'])
        self.assertIn('projected_support',result['pixel_regions'])
        self.assertIn('udf_relative_mae_mm',result['geometry_regions']['all'])
        self.assertIn('corrected_minus_predicted_gain_ci',result)
        with self.assertRaises(FileExistsError):diagnose(a)


if __name__=='__main__': unittest.main()
