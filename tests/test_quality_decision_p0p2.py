"""Synthetic CPU contract tests for P0/P1/P2. No CAD or CUDA claims."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from rare_grasp.core import physical_ids, validate_cache
from diagnostics.quality_decision.core import (
    QualityProbe, action_geometry, digest, frame_metrics, predictor,
    objective, scene_equal_aggregate,
)
from diagnostics.quality_decision.fit import select_by_scene
from diagnostics.quality_decision.mine import choose_queries
from diagnostics.quality_decision.global_budget import (
    global_greedy_quality, exact_ap_rank_weights, verify_non_conflicting,
    analyze_frame,
)


def synthetic_actions(n_family=2):
    n=n_family*4
    g=np.zeros((n,17),dtype=np.float32)
    g[:,1]=.06;g[:,2]=.02;g[:,3]=np.tile(np.array([.01,.02,.03,.04]),n_family)
    g[:,4:13]=np.eye(3).reshape(9);g[:,13]=np.repeat(np.arange(n_family)*.05,4)
    g[:,15]=.5;g[:,16]=-1
    return g


def cache():
    grasp=synthetic_actions()
    count=np.array([0,1,5,6,6,4,2,0],dtype=np.int64)
    np.random.seed(5)
    return dict(grasps=grasp,physical_id=physical_ids(grasp),
                features=np.random.standard_normal((8,64)).astype(np.float32),
                base_logit=np.linspace(-1,1,8).astype(np.float32),
                depth=np.tile(np.arange(4),2),family=np.repeat([0,1],4),
                count=count,
                bits=np.array([(np.arange(6)>=(6-c)).astype(np.uint8) for c in count]))


class P0Tests(unittest.TestCase):
    def test_cache_and_zero_init_parity_all_modes(self):
        x=cache();validate_cache(x)
        base=torch.as_tensor(x["base_logit"])
        for mode in ("score","score_geometry","score_feature",
                     "score_feature_geometry","score_feature_permuted"):
            with self.subTest(mode=mode):
                torch.manual_seed(13)
                m=QualityProbe(mode)
                out=predictor(m,x,torch.device("cpu"),seed=1)
                torch.testing.assert_close(out,base,atol=0,rtol=0)
                self.assertTrue(torch.isfinite(out).all())

    def test_mode_structure_and_matched_capacity(self):
        names=("score","score_geometry","score_feature","score_feature_geometry")
        params=[sum(p.numel() for p in QualityProbe(m).parameters()) for m in names]
        self.assertEqual(len(set(params)),1)
        self.assertGreater(params[0],1000)

    def test_action_geometry_scaling(self):
        g=torch.as_tensor(synthetic_actions(1))
        z=action_geometry(g)
        self.assertEqual(tuple(z.shape),(4,15))
        self.assertAlmostEqual(float(z[0,0]),.6)
        self.assertAlmostEqual(float(z[2,2]),.75)

    def test_frame_metric_regret_and_pairwise(self):
        x=cache();scores=np.array([0.1,0.2,0.3,0.99,0.9,0.7,0.4,0.1])
        m=frame_metrics(scores,x)
        self.assertAlmostEqual(m["selected_q"],1)
        self.assertAlmostEqual(m["oracle_q"],1)
        self.assertAlmostEqual(m["regret"],0)
        self.assertEqual(m["actions"],8)

    def test_loss_non_ties_and_zero_point_grad(self):
        x=cache()
        z=torch.tensor(x["base_logit"],requires_grad=True)
        c=torch.as_tensor(x["count"])
        f=torch.as_tensor(x["family"])
        loss,audit=objective(z,c,f,pair_weight=.5,max_pairs=20,seed=0)
        self.assertTrue(float(loss)>0)
        self.assertGreater(audit["used_pairs"],0)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_scene_equal_aggregate(self):
        rows=[
            dict(domain="original",scene=1,mse=0.,selected_q=0.,oracle_q=1.,regret=1.,pairwise=.2),
            dict(domain="original",scene=1,mse=0.,selected_q=0.,oracle_q=1.,regret=1.,pairwise=.2),
            dict(domain="original",scene=2,mse=1.,selected_q=1.,oracle_q=1.,regret=0.,pairwise=.8),
            dict(domain="gntrans",scene=1,mse=1.,selected_q=1.,oracle_q=1.,regret=0.,pairwise=.8),
        ]
        result=scene_equal_aggregate(rows)
        self.assertAlmostEqual(result["mse"],.75)


class P1Tests(unittest.TestCase):
    def test_scene_held_out_selection(self):
        entries=[]
        for domain in ("original","gntrans"):
            for scene in range(3):
                for anno in range(10):
                    entries.append(dict(row=dict(domain=domain,scene=scene,anno=anno)))
        train=select_by_scene(entries,0,2,3)
        val=select_by_scene(entries,2,3,3)
        self.assertEqual(len(train),12)
        self.assertEqual(len(val),6)
        self.assertFalse({e["row"]["scene"] for e in train}&{e["row"]["scene"] for e in val})

    def test_mining_budgets_and_feature_alignment(self):
        bank=np.zeros((1024,4,17),dtype=np.float32)
        bank[...,1]=.06;bank[...,2]=.02
        bank[...,3]=np.arange(1,5)[None,:]*.01
        bank[...,4:13]=np.eye(3).reshape(9)
        bank[...,13]=np.arange(1024)[:,None]*.004
        bank[...,15]=.5;bank[...,16]=-1
        bank[...,0]=np.linspace(.1,.9,1024)[:,None]
        snap=dict(grasps=bank)
        class Fake:
            def sensor_collision_filter(self,row,actions):
                return actions[:24].copy()
        uni,decision,audit=choose_queries(snap,Fake(),dict(
            domain="original",scene=0,anno=0,split="train",partition="train"),
            seed=2)
        self.assertEqual(len(uni),64)
        self.assertEqual(len(decision),64)
        self.assertEqual(len(np.unique(decision)),64)
        self.assertGreaterEqual(len(set(map(int,uni))&set(map(int,decision))),32)
        self.assertEqual(audit["post_collision"],24)


class P2Tests(unittest.TestCase):
    def test_greedy_global_budget_and_independence(self):
        edges=np.array([[0,1],[1,2]],dtype=np.int64)
        scores=np.array([.5,.9,.3,.1],dtype=float)
        quality=np.array([6,1,6,2],dtype=np.int64)
        obj=np.array([0,0,0,1])
        selected=global_greedy_quality(np.arange(4),edges,quality,obj,scores,top_k=2)
        self.assertEqual(len(selected),2)
        verify_non_conflicting(selected,edges,obj,k=2)
        self.assertNotIn(1,selected.tolist())

    def test_global_rank_ap_weight_monotonic(self):
        weights=exact_ap_rank_weights()
        self.assertEqual(len(weights),50)
        self.assertTrue(np.all(np.diff(weights)<0))
        self.assertAlmostEqual(weights[-1],(1/50)/(50*6))
        self.assertAlmostEqual(weights[0],sum(1/k for k in range(1,51))/(50*6))

    def test_budget_one_synthetic_full_frame(self):
        g=synthetic_actions(2)
        g[:,0]=np.linspace(.9,.2,8)
        ids=physical_ids(g)
        edges=np.array([[0,1],[1,2],[4,5]],dtype=np.int64)
        from diagnostics.candidate_conflict.core import connected_components,greedy_nms
        comp=connected_components(len(g),edges)
        keep,suppressor,order=greedy_nms(g[:,0],edges)
        graph=dict(grasps=g,physical_id=ids,edges=edges,
                   component_id=comp,nms_keep=keep,suppressor=suppressor,nms_order=order)
        counts=np.array([0,6,6,0,1,5,0,4],dtype=np.uint8)
        bits=np.asarray([(np.arange(6)>=(6-int(c))).astype(np.uint8) for c in counts])
        labels=dict(physical_id=ids,assigned_obj=np.array([0]*4+[1]*4),
                    collision_or_empty=np.zeros(8,dtype=bool),
                    friction=np.array([-1,.2,.2,-1,1.2,.4,-1,.6]),
                    success_bits=bits,quality_count=counts)
        result=analyze_frame(graph,labels,solver="off")
        self.assertTrue(0<=result["native_ap_percent"]<=100)
        self.assertTrue(0<=result["global_greedy_ap_percent"]<=100)
        self.assertEqual(result["num_candidates"],8)


if __name__=="__main__":
    unittest.main()
