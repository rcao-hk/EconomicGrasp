"""Synthetic CPU contracts for P0/P1/P2. Real CUDA/CAD tests are separate."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from diagnostics.action_decision.model import (
    MODES,ActionQualityProbe,action_geometry,make_input,choose_pairs,quality_losses,
)
from diagnostics.action_decision.feature_cache import gather_reader
from diagnostics.action_decision.global_oracle import (
    restricted_pool,feasible_quality_greedy,solve_set,
)
from diagnostics.candidate_conflict.core import build_conflict_edges


def make_actions(families=4):
    n=families*4
    grasp=np.zeros((n,17),dtype=np.float32)
    grasp[:,1]=0.06;grasp[:,2]=0.02
    grasp[:,3]=np.tile(np.arange(1,5)*.01,families)
    grasp[:,4:13]=np.eye(3,dtype=np.float32).ravel()
    grasp[:,13]=np.repeat(np.arange(families)*.1,4)
    grasp[:,15]=.5;grasp[:,16]=-1
    return grasp


def data(families=4):
    from rare_grasp.core import physical_ids
    g=make_actions(families);n=len(g)
    source=dict(features=np.arange(n*64,dtype=np.float32).reshape(n,64)/1000,
                base_logit=np.linspace(-1,1,n,dtype=np.float32),
                depth=np.tile(np.arange(4),families),
                family=np.repeat(np.arange(families),4),
                count=np.tile(np.array([0,1,3,6]),families),
                grasps=g,physical_id=physical_ids(g),
                native_angle=np.zeros(n,dtype=np.int64))
    extra=dict(reader=np.ones((n,256),np.float16),
               reader_delta=np.linspace(-1,1,n*256,dtype=np.float32).reshape(n,256),
               depth=source["depth"],family=source["family"],
               physical_id=source["physical_id"])
    return source,extra


class P0Tests(unittest.TestCase):
    def test_geometry_is_actual_physical_action_only(self):
        g=make_actions();x=action_geometry(g)
        self.assertEqual(x.shape,(16,15))
        np.testing.assert_allclose(x[:,0],.6)
        np.testing.assert_allclose(x[:,1],1.)
        g[:,0]=1000
        np.testing.assert_allclose(action_geometry(g),x)

    def test_all_probes_epoch_zero_native_parity_and_capacity_match(self):
        source,reader=data()
        inputs=make_input(source,reader,"cpu")
        dims=set()
        for mode in MODES:
            model=ActionQualityProbe(mode,hidden=32)
            with torch.no_grad():
                z=model(inputs)
            torch.testing.assert_close(z,inputs["base"],atol=0,rtol=0)
            dims.add(sum(p.numel() for p in model.parameters()))
        self.assertEqual(len(dims),1)

    def test_reader_action_identity_required(self):
        source,reader=data()
        reader={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in reader.items()}
        reader["physical_id"][0]="wrong"
        with self.assertRaises(RuntimeError):
            make_input(source,reader,"cpu")

    def test_gather_reader_selects_native_query_angle_depth(self):
        source,_=data(families=2)
        before=torch.zeros((1,1024,12,4,256))
        after=before.clone()
        after[0,1,3,2,0]=5.
        snap=dict(angles=np.zeros(1024,dtype=np.int64),
                  grasps=np.zeros((1024,4,17),dtype=np.float32),
                  features=np.zeros((1024,4,64),dtype=np.float32),
                  base_logit=np.zeros((1024,4),dtype=np.float32))
        q=source["family"];d=source["depth"]
        snap["grasps"][q,d]=source["grasps"]
        snap["features"][q,d]=source["features"]
        snap["base_logit"][q,d]=source["base_logit"]
        result=gather_reader((before,after),snap,source)
        self.assertEqual(result["reader"].shape,(8,256))
        self.assertEqual(float(result["reader"].sum()),0.)
        snap["angles"][1]=3
        source["native_angle"][q==1]=3
        result=gather_reader((before,after),snap,source)
        i=np.flatnonzero((q==1)&(d==2))[0]
        self.assertEqual(float(result["reader"][i,0]),5.0)

    def test_quality_loss_ignores_equal_quality_pairs(self):
        source,extra=data()
        inp=make_input(source,extra,"cpu")
        scores=torch.zeros(16,requires_grad=True)
        mse,pair,stats=quality_losses(scores,inp,"rare",1000,0)
        self.assertGreater(stats["used"],0)
        self.assertTrue(torch.isfinite(mse+pair))
        pair.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

class P1Tests(unittest.TestCase):
    def test_pair_selection_deterministic_and_quality_non_ties(self):
        source,_=data()
        a,sa=choose_pairs(source,"decision",budget=10,seed=15,top_k=3)
        b,sb=choose_pairs(source,"decision",budget=10,seed=15,top_k=3)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(sa,sb)
        self.assertTrue(all(source["count"][i]!=source["count"][j] for i,j in a))

    def test_objective_inputs_same_labels(self):
        source,reader=data()
        inp=make_input(source,reader,"cpu")
        model=ActionQualityProbe("reader256",32)
        z=model(inp)
        for objective in ("mse","rare","decision"):
            mse,pair,_=quality_losses(z,inp,objective,100,1)
            self.assertTrue(torch.isfinite(mse+pair))

    def test_conflict_graph_not_equivalent_to_connected_component_cluster(self):
        g=make_actions(3)[::4]
        g[:,13]=[0.,.02,.04]
        edges=build_conflict_edges(g)
        self.assertEqual(edges.tolist(),[[0,1],[1,2]])

class P2Tests(unittest.TestCase):
    def test_restricted_pool_retains_native_survivors(self):
        q=np.array([1,6,2,0,4,1])
        s=np.array([.9,.8,.7,.6,.5,.4])
        indices,restricted=restricted_pool(q,s,[0,2],2)
        self.assertTrue(restricted)
        self.assertTrue({0,2}.issubset(set(indices.tolist())))

    def test_feasible_greedy_global_budget_and_object_cap(self):
        q=np.array([6,5,4,3])
        s=np.array([.8,.9,.7,.6])
        obj=np.array([1,1,1,2])
        edges=np.array([[0,1],[1,2]],dtype=np.int64)
        selected=feasible_quality_greedy(q,s,edges,obj,2,1)
        self.assertLessEqual(len(selected),2)
        self.assertLessEqual(int(np.sum(obj[selected]==1)),1)
        self.assertFalse(0 in selected and 1 in selected)

    def test_cp_sat_global_selection_if_available(self):
        try:from ortools.sat.python import cp_model
        except ImportError:self.skipTest("ortools optional for CPU tests")
        q=np.array([6,1,5,3])
        s=np.array([.8,.9,.7,.6])
        obj=np.array([0,0,1,1])
        edges=np.array([[0,1],[1,2]],dtype=np.int64)
        picked,stat=solve_set(q,s,edges,obj,budget=2,per_object=1,
                              time_limit=5.,workers=1,max_candidates=0)
        self.assertEqual(stat["status"],"OPTIMAL")
        self.assertTrue(stat["certifies_full_pool_sum_q"])
        self.assertEqual(int(q[picked].sum()),11)
        self.assertTrue(set(picked.tolist())=={0,2})

    def test_no_exactness_claim_when_pool_restricted(self):
        try:from ortools.sat.python import cp_model
        except ImportError:self.skipTest("ortools optional for CPU tests")
        q=np.array([6,5,4,3])
        s=np.array([.9,.8,.7,.6])
        obj=np.arange(4)
        edges=np.array([[0,1]],np.int64)
        _,stats=solve_set(q,s,edges,obj,budget=2,per_object=1,
                          max_candidates=1,time_limit=5.)
        self.assertTrue(stats["restricted"])
        self.assertFalse(stats["certifies_full_pool_sum_q"])


if __name__=="__main__":
    unittest.main()
