"""Pure CPU contracts for Height P0 A/B/C. No full model/checkpoint required."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import json

import numpy as np
import torch

from diagnostics.height_p0.metrics import (
    deterministic_frames,scene_ids,frame_metrics,paired_appearance_metrics,
    grouped_ci,object_pool,aggregate,
)
from diagnostics.height_p0.checkpoints import inspect_checkpoint
from tools.height_p0_abc import ray_dot_from_K,read_prior,SCHEMA


class SelectionTests(unittest.TestCase):
    def test_train_used_is_even_and_holdout_odd(self):
        a=deterministic_frames("train_used",128)
        b=deterministic_frames("train_heldout_view",128)
        self.assertEqual(a,list(range(0,256,2)))
        self.assertEqual(b,list(range(1,256,2)))
        self.assertFalse(set(a)&set(b))
        self.assertEqual(len(scene_ids("test_novel")),30)

    def test_scene_partition_disjoint(self):
        groups=[set(scene_ids(k)) for k in ("train_used","test_seen","test_similar","test_novel")]
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                self.assertFalse(groups[i]&groups[j])

    def test_illegal_fraction_rejected(self):
        with self.assertRaises(ValueError):deterministic_frames("train_used",129)

    def test_scene_balanced_bootstrap(self):
        rows=[{"scene":"a","loss":1.} for _ in range(50)]+[{"scene":"b","loss":5.}]
        self.assertAlmostEqual(grouped_ci(rows,"loss",50)["mean"],3.)


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.gt=np.array([[.4,.5,.6,.7],[.45,.55,.65,.75]],dtype=np.float32)
        self.dot=np.full_like(self.gt,-1.)
        self.seg=np.array([[1,1,2,2],[1,1,2,2]],dtype=np.int32)
        self.b=.85

    def test_height_depth_same_physics(self):
        z=self.gt+.01
        result,objs,mask,h=frame_metrics(
            z,self.gt,self.dot,self.b,self.seg,(-.05,.35),
            expected_height=self.b-z,prior_h=.25)
        self.assertAlmostEqual(result["z_mae_mm"],10.,places=4)
        self.assertAlmostEqual(result["h_mae_mm"],10.,places=4)
        self.assertEqual(len(objs),2)
        self.assertLess(result["height_roundtrip_max_mm"],1e-3)
        self.assertTrue(mask.all())
        self.assertAlmostEqual(result["pred_invalid_depth_fraction"],0.)

    def test_predicted_negative_depth_is_not_dropped(self):
        z=-self.gt
        result,_,mask,_=frame_metrics(z,self.gt,self.dot,self.b,self.seg,(-.05,.35))
        self.assertTrue(mask.all())
        self.assertGreater(result["z_mae_mm"],400.)
        self.assertEqual(result["pred_invalid_depth_fraction"],1.)

    def test_invalid_GT_and_rays_are_same_for_both_models(self):
        z0=self.gt+.1;z1=self.gt-.2
        g=self.gt.copy();g[0,0]=0
        dot=self.dot.copy();dot[0,1]=-.01
        _,_,ma,_=frame_metrics(z0,g,dot,self.b,self.seg,(-.05,.35))
        _,_,mb,_=frame_metrics(z1,g,dot,self.b,self.seg,(-.05,.35))
        self.assertTrue(np.array_equal(ma,mb))
        self.assertEqual(int(ma.sum()),6)

    def test_height_range_outlier_reported_without_filtering(self):
        z=self.gt+.01
        res,_,mask,_=frame_metrics(z,self.gt,self.dot,self.b,self.seg,(-.05,.35))
        self.assertGreater(res["gt_height_outside_head_range_fraction"],0.)
        self.assertEqual(mask.sum(),8)

    def test_wrong_analytic_roundtrip_rejected(self):
        with self.assertRaises(ValueError):
            frame_metrics(self.gt,self.gt,self.dot,self.b,self.seg,(-.05,.35),
                          expected_height=np.zeros_like(self.gt))

    def test_pair_same_geometry_and_GT(self):
        p=paired_appearance_metrics(self.gt,self.gt+.01,self.gt,self.dot,
                                    np.ones_like(self.gt,dtype=bool))
        self.assertAlmostEqual(p["appearance_z_delta_mae_mm"],10.,places=4)
        self.assertAlmostEqual(p["trans_minus_original_z_mae_mm"],10.,places=4)
        self.assertEqual(p["trans_worse_pixel_fraction"],1.)
        with self.assertRaises(ValueError):
            paired_appearance_metrics(self.gt,self.gt[:1],self.gt,self.dot,
                                      np.ones_like(self.gt,dtype=bool))

    def test_cross_object_stats_exclude_scene_collision(self):
        z=self.gt+.01
        _,o,_,_=frame_metrics(z,self.gt,self.dot,self.b,self.seg,(-.05,.35))
        rows=[]
        for sc in ("scene_0001","scene_0002"):
            rows.append({"family":"height_analytic","epoch":16,"split":"test_similar",
                         "domain":"original","scene":sc,"frame":0,"objects":o})
        pooled=object_pool(rows)
        self.assertEqual(len(pooled),4)
        self.assertTrue(all(abs(x["h_mae_mm"]-10)<1e-3 for x in pooled))


class ProtocolTests(unittest.TestCase):
    def fixture_ck(self):
        return {"epoch":16,"pose_depth_mode":"height_analytic",
            "grasp_design":{"enable_point_rayrope":True,"enable_gvar":True},
            "gntrans_mix_protocol":{"train_fraction_each_domain":.5,
                "original_train_count":12800,"gntrans_train_count":12800},
            "height_analytic_protocol":{"mode":"height_analytic",
                "no_pose_ray_plane_film":True,
                "height_min_m":-.05,"height_max_m":.35,
                "height_min_downward_dot":.1},
            "model_state_dict":{
                "depth_net.height_analytic_signature":torch.tensor([-.05,.35,.1]),
                "depth_net.depthnet.fake_weight":torch.tensor([1.]),
                "some_grasp_head.weight":torch.tensor([0.]),
            }}

    def test_height_checkpoint_accepts_exact_protocol(self):
        state,audit=inspect_checkpoint(self.fixture_ck(),"height_analytic",16)
        self.assertIn("height_analytic_signature",state)

    def test_mismatched_epoch_and_design_rejected(self):
        ck=self.fixture_ck()
        with self.assertRaises(ValueError):inspect_checkpoint(ck,"height_analytic",15)
        ck["grasp_design"]["enable_gvar"]=False
        with self.assertRaises(ValueError):inspect_checkpoint(ck,"height_analytic",16)

    def test_pose_film_cannot_masquerade_as_height(self):
        ck=self.fixture_ck()
        ck["model_state_dict"]["depth_net.pose_aware_adapter.weight"]=torch.ones(1)
        with self.assertRaises(ValueError):inspect_checkpoint(ck,"height_analytic",16)

    def test_train_only_constant_prior_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"prior.json"
            path.write_text(json.dumps(dict(schema=SCHEMA,partition="train_used",
                dataset_root=tmp,gt_source="rendered_object_foreground",h0_m=.15)))
            h,_=read_prior(path,tmp);self.assertAlmostEqual(h,.15)
            p=json.loads(path.read_text());p["partition"]="test_similar"
            path.write_text(json.dumps(p))
            with self.assertRaises(ValueError):read_prior(path,tmp)

    def test_pixel_ray_matrix(self):
        K=np.array([[2.,0.,2.],[0.,2.,2.],[0.,0.,1.]])
        p=np.array([0.,0.,-1.,.8])
        dot=ray_dot_from_K(K,p,(5,5))
        np.testing.assert_allclose(dot,-1.)

    def test_generalization_gap_and_paired_diff_in_gap(self):
        base=dict(domain="original",frame=0,n_common_pixels=2,
                  paired=None,objects=[])
        rows=[]
        for family,train_error,similar_error in (
            ("height_analytic",1.,4.),("global_film",2.,3.)):
            for split,scene,value in (("train_used","scene_0000",train_error),
                                      ("test_similar","scene_0132",similar_error)):
                metrics={"z_mae_mm":value,"h_mae_mm":value,
                         "z_p95_abs_mm":value+1.}
                rows.append(dict(base,family=family,epoch=16,split=split,
                                 scene=scene,metrics=metrics))
        agg=aggregate(rows,expected_epochs=[16],n_boot=100)
        k="height_analytic/epoch16/original/test_similar/z_mae_mm"
        self.assertAlmostEqual(agg["generalization_gaps"][k]["gap"],3.)
        j="epoch16/original/test_similar/z_mae_mm"
        self.assertAlmostEqual(agg["paired_generalization_gap_differences"][j]["gap"],2.)

    def test_unpaired_epochs_not_silently_interpolated(self):
        base=dict(split="test_similar",domain="original",scene="scene_0132",frame=0,
                  n_common_pixels=2,metrics={"z_mae_mm":10.,"h_mae_mm":8.,
                                              "z_p95_abs_mm":20.},
                  paired=None,objects=[])
        a=dict(base,family="height_analytic",epoch=16)
        b=dict(base,family="global_film",epoch=12)
        r=aggregate([a,b],expected_epochs=[12,16],n_boot=20)
        self.assertFalse(r["matched_epoch_contrasts"])
        self.assertEqual(r["missing_expected_epochs"]["height_analytic"],[12])


if __name__=="__main__":
    unittest.main()
