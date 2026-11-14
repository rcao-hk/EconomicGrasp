"""Synthetic CPU contracts for non-training P0/P1 follow-up diagnostics.

Tests intentionally do NOT claim real GraspNet CAD parity or 117 GPU coverage.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from diagnostics.decision_supervision.core import BASE_SHA, physical_ids
from diagnostics.decision_supervision.attribution_core import (
    common_bank, compiled_keep, final_ids, arrange_labels, replay,
)
from diagnostics.decision_supervision.support_audit import (
    _native_known, _sample_new, _selected, _frame,
)


def make_grasps(n=5):
    g = np.zeros((n, 17), dtype=np.float64)
    g[:, 0] = np.linspace(.95,.5,n)
    g[:, 1] = .06;g[:, 2] = .02;g[:, 3] = .02
    g[:,4:13] = np.eye(3).reshape(9)
    g[:,13] = np.arange(n)*.1
    if n > 1:g[1,13] = .01
    g[:,15] = .5;g[:,16] = -1
    return g


def fake_nms(rows):
    g = np.asarray(rows)
    order=np.argsort(g[:,0])[::-1]
    seen=set();keep=[]
    for ix in order:
        t=float(g[ix,13])
        if any(abs(t-float(g[j,13]))<.03 for j in keep):
            continue
        keep.append(int(ix))
    return g[keep].copy()


class P0Tests(unittest.TestCase):
    def test_compiled_nms_preserves_rows(self):
        g = make_grasps()
        keep=compiled_keep(g,fake_nms)
        self.assertEqual(keep.tolist(),[0,2,3,4])

    def test_common_bank_rejects_geometry_drift(self):
        a=make_grasps()
        b=a.copy();b[:,0] = b[:,0][::-1]
        common_bank({"A":a,"C":b})
        b[0,1]=.07
        with self.assertRaisesRegex(ValueError,"Physical candidates differ"):
            common_bank({"A":a,"C":b})

    def test_official_rank_selection_can_include_score_ties(self):
        scores=np.array([1.,.9,.9,.8,.7])
        keep=np.arange(5,dtype=np.int64)
        obj=np.array([0,0,0,1,1])
        output=final_ids(keep,scores,obj,per_object=2,k=3)
        self.assertEqual(set(output.tolist()),{0,1,3})
        self.assertEqual(len(output),3)

    def test_exact_label_mapping(self):
        ids=np.array([0,2,4])
        ev=SimpleNamespace(assigned_obj=np.array([0,1,1]),
             collision_or_empty=np.array([False,False,True]),
             friction=np.array([.2,.6,-1.]))
        assigned,bad,friction=arrange_labels(5,ids,ev)
        self.assertEqual(assigned[2],1)
        self.assertEqual(friction[3],-1)
        self.assertTrue(bad[4])

    def test_pathwise_stages_sum_and_detect_q_inversion(self):
        g=make_grasps(5)
        score_a=np.array([.95,.85,.7,.6,.5])
        score_c=np.array([.1,.99,.7,.6,.5])
        a=g.copy();a[:,0]=score_a
        c=g.copy();c[:,0]=score_c
        ka=compiled_keep(a,fake_nms)
        kc=compiled_keep(c,fake_nms)
        assert set(ka)!={*kc}
        assigned=np.array([0,0,1,1,2])
        bad=np.zeros(5,dtype=bool)
        friction=np.array([.2,-1.,.4,.8,1.2])
        result=replay(g,score_a,score_c,ka,kc,assigned,bad,friction)
        self.assertAlmostEqual(
            result["total_pp"],
            result["rank_fixed_final_pp"]+result["quota_reselection_pp"]+
            result["nms_set_replacement_pp"],places=9)
        self.assertGreater(result["positive_lost_zero_gained_score_inversions"],0)
        self.assertTrue(result["final_lost"]>=1)

    def test_official_ap_parity_is_a_hard_gate(self):
        g=make_grasps(5)
        a=np.array([.95,.85,.7,.6,.5]);b=np.array([.1,.99,.7,.6,.5])
        ka=compiled_keep(np.column_stack((a,g[:,1:])),fake_nms)
        kb=compiled_keep(np.column_stack((b,g[:,1:])),fake_nms)
        with self.assertRaisesRegex(RuntimeError,"AP parity failed"):
            replay(g,a,b,ka,kb,np.array([0,0,1,1,2]),
                   np.zeros(5,dtype=bool),np.array([.2,-1.,.4,.8,1.2]),
                   official_left=np.ones((50,6)))

    def test_mismatched_ap_missing_frictions_fails(self):
        with self.assertRaises(ValueError):
            arrange_labels(5,np.array([0,1]),
                 SimpleNamespace(assigned_obj=[0],collision_or_empty=[False,False],
                                 friction=[.2,.4]))


class P1Tests(unittest.TestCase):
    def test_sample_allocation_same_seed_and_no_old_duplicates(self):
        eligible=np.arange(200,dtype=np.int64)
        scores=np.linspace(1.,0.,200)
        keep=np.arange(0,50)
        edges=np.array([[0,90],[1,91],[2,92],[3,93],[4,94]],dtype=np.int64)
        old={i:dict(count=0) for i in range(0,60)}
        a,labels,audit,leaders=_sample_new(eligible,scores,keep,edges,old=old,seed=19)
        b,bl,ba,lead=_sample_new(eligible,scores,keep,edges,old=old,seed=19)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(len(a),64)
        self.assertEqual(len(np.unique(a)),64)
        self.assertFalse(any(int(i) in old for i in a))
        self.assertEqual(sum(audit["sample_count_by_stratum"].values()),64)
        self.assertEqual(audit["top50_cached_n"],50)
        self.assertEqual(audit["competition_nodes_n"],5)

    def test_audit_does_not_read_quality_for_sampling(self):
        eligible=np.arange(300,dtype=np.int64)
        scores=np.linspace(0,1,300)
        keep=np.array([295,296,297,298,299])
        result1=_sample_new(eligible,scores,keep,np.empty((0,2),int),old={1:{"count":0}},seed=7)[0]
        result2=_sample_new(eligible,scores,keep,np.empty((0,2),int),old={1:{"count":6}},seed=7)[0]
        np.testing.assert_array_equal(result1,result2)

    def test_strata_shortfall_reported_not_fabricated(self):
        eligible=np.arange(75)
        result,labels,audit,_=_sample_new(
            eligible,np.arange(75,dtype=float),np.arange(50),
            np.empty((0,2),dtype=int),old=set(range(20)),seed=9)
        self.assertEqual(len(result),55)
        self.assertGreater(audit["fallback_n"],0)
        self.assertEqual(audit["new_labeled_n"],55)
        self.assertGreater(audit["quota_deficits"]["new_top50_conflict_neighbor"],0)

    def test_scene_sampling_never_accepts_test_scenes(self):
        entries=[]
        for domain in ("original","gntrans"):
            for scene in range(100):
                for anno in (0,2,4,6):
                    entries.append(dict(row=dict(domain=domain,scene=scene,anno=anno,split="train")))
        result=_selected({"entries":entries},("original","gntrans"),2)
        self.assertEqual(len(result),400)
        self.assertEqual(len({(x["row"]["domain"],x["row"]["scene"]) for x in result}),200)
        entries[0]["row"]["scene"]=105
        with self.assertRaises(ValueError):
            _selected({"entries":entries},("original","gntrans"),2)

    def test_resume_does_not_label_again(self):
        # Minimal faithful snapshot of 1024 queries and a previously labeled
        # U∪D cache. Production has stricter replays and compiled NMS.
        bank=np.zeros((1024,4,17),dtype=np.float64)
        bank[...,0]=np.linspace(.2,.9,1024)[:,None]
        bank[...,1]=.06;bank[...,2]=.02
        bank[...,3]=np.arange(1,5)[None,:]*.01
        bank[...,4:13]=np.eye(3).reshape(9)
        bank[...,13]=np.arange(1024)[:,None]*.005
        bank[...,15]=.5;bank[...,16]=-1
        fam=np.repeat(np.arange(96),4)
        dep=np.tile(np.arange(4),96)
        x=dict(family=fam,depth=dep,
               grasps=bank[fam,dep].copy(),
               physical_id=physical_ids(bank[fam,dep]),
               features=np.zeros((len(fam),64),np.float32),
               base_logit=np.zeros(len(fam),np.float32),
               count=np.zeros(len(fam),np.int64),
               native_sensor_valid=np.ones(96,dtype=bool))
        class FakeRT:
            label_calls=0
            def snapshot(self,row,seed):
                return dict(grasps=bank,features=np.zeros((1024,4,64),np.float32),
                            base_logit=np.zeros((1024,4),np.float32))
            def sensor_collision_filter(self,row,grasps):
                return grasps.copy()
            def label(self,row,grasps):
                self.label_calls+=1;n=len(grasps)
                return dict(count=np.zeros(n,np.int64),bits=np.zeros((n,6),np.uint8),
                            friction=np.full(n,-1.,dtype=np.float64),
                            collision=np.zeros(n,np.uint8),empty=np.zeros(n,np.uint8))
        rt=FakeRT()
        row=dict(domain="original",scene=0,anno=0,split="train")
        entry=dict(row=row,sha256="fake-cached-pack-hash")
        args=SimpleNamespace(snapshot_seed=123,seed=8,min_new_actions=1,
                             max_compiled_pair_checks=0)
        contract=dict(plan_sha256="plan",script_sha256="script")
        with tempfile.TemporaryDirectory() as td:
            with patch("diagnostics.decision_supervision.support_audit.read_pack",return_value=x), \
                 patch("diagnostics.decision_supervision.support_audit.compiled_keep",
                       return_value=np.arange(50,dtype=np.int64)), \
                 patch("diagnostics.decision_supervision.support_audit.build_conflict_edges",
                       return_value=np.empty((0,2),np.int64)):
                first=_frame(args,rt,fake_nms,entry,Path(td),contract,False)
                self.assertEqual(rt.label_calls,1)
                second=_frame(args,rt,fake_nms,entry,Path(td),contract,True)
                self.assertEqual(rt.label_calls,1)
                self.assertEqual(first["sha256"],second["sha256"])


if __name__ == "__main__":
    unittest.main()
