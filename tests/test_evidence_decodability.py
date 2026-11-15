"""CPU synthetic tests for frozen-GVAR P0 matching and P1 evidence probes.

These tests do not establish parity with 117's CAD evaluator or compiled NMS.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from diagnostics.decision_supervision.core import physical_ids
from diagnostics.evidence_decodability.core import (
    EvidenceProbe, MODES, check_bank, failure_class,
    failure_metrics, quality_stats,
)
from diagnostics.evidence_decodability.capture import GVARCapture
from diagnostics.evidence_decodability.train import _frame_subset
from diagnostics.evidence_decodability.p0_match import _inversion_counts
from diagnostics.evidence_decodability.cache import _choose_frames, _join_and_validate, _old_native
from diagnostics.evidence_decodability.infer import native_actions
from diagnostics.evidence_decodability.core import sha256

ROOT=Path(__file__).resolve().parents[1]


def sample(n=128,H=32):
    g=np.zeros((n,17),dtype=np.float32)
    g[:,0]=np.linspace(.4,.9,n)
    g[:,1]=.06;g[:,2]=.02;g[:,3]=.02
    g[:,4:13]=np.eye(3,dtype=np.float32).reshape(9)
    g[:,13]=np.arange(n)*.002
    g[:,15]=.7;g[:,16]=-1
    count=np.arange(n,dtype=np.int64)%7
    collision=(count==0)&(np.arange(n)%3==0)
    empty=(count==0)&(np.arange(n)%3==1)
    bits=np.asarray([(np.arange(6)>=6-int(q)).astype(np.uint8) for q in count])
    rng=np.random.default_rng(3)
    return dict(
        query_id=np.arange(n,dtype=np.int64),
        depth_id=np.full(n,1,dtype=np.int64),
        native_score=g[:,0].copy(),
        base_logit=np.log(g[:,0]/(1-g[:,0])).astype(np.float32),
        grasps=g,physical_id=physical_ids(g),source_new=np.r_[
            np.zeros(n//2,dtype=np.uint8),np.ones(n-n//2,dtype=np.uint8)],
        sensor_valid=np.ones(n,dtype=np.uint8),
        nms_role=np.resize([0,1,2],n).astype(np.int8),
        native_score_percentile=np.linspace(.1,1,n).astype(np.float32),
        quality_count=count,quality_bits=bits,
        pure_collision=collision.astype(np.uint8),empty=empty.astype(np.uint8),
        score64=rng.standard_normal((n,64)).astype(np.float32),
        pre_reader=rng.standard_normal((n,H)).astype(np.float32),
        post_reader=rng.standard_normal((n,H)).astype(np.float32),
        reader_delta=np.ones((n,H),dtype=np.float32),
        role_tokens=rng.standard_normal((n,6,64)).astype(np.float32),
        role_mass=np.full((n,6),1/6,dtype=np.float32),
        role_valid=np.ones((n,6),dtype=np.float32),
        pred_geometry=rng.standard_normal((n,6,6)).astype(np.float32),
    )


class CoreTests(unittest.TestCase):
    def test_all_evidence_modes_same_parameters_and_zero_quality_init(self):
        x=sample()
        h=check_bank(x)
        shapes=[]
        for mode in MODES:
            if mode=="reference_geometry":
                x["reference_geometry"]=x["pred_geometry"].copy()
            model=EvidenceProbe(h,mode,"quality",width=32)
            shapes.append(sum(p.numel() for p in model.parameters()))
            with torch.inference_mode():
                logits=model(x)
            self.assertEqual(tuple(logits.shape),(len(x["query_id"]),))
            np.testing.assert_allclose(logits.numpy(),x["base_logit"],atol=0,rtol=0)
        self.assertEqual(len(set(shapes)),1)

    def test_failure_probe_logits_and_gradient(self):
        x=sample()
        model=EvidenceProbe(32,"role_tokens","failure",width=32)
        prediction=model(x)
        self.assertEqual(tuple(prediction.shape),(128,4))
        target=torch.as_tensor(failure_class(x["quality_count"],
                                             x["pure_collision"],x["empty"]))
        loss=nn.CrossEntropyLoss()(prediction,target)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all()
                            for p in model.parameters() if p.grad is not None))

    def test_failure_labels_priority(self):
        q=np.array([6,0,0,0,0])
        coll=np.array([0,1,0,1,0])
        empty=np.array([0,1,1,0,0])
        np.testing.assert_array_equal(failure_class(q,coll,empty),[0,1,2,1,3])
        with self.assertRaises(ValueError):
            failure_class(np.array([2]),np.array([1]),np.array([0]))

    def test_physical_action_label_mismatch_is_rejected(self):
        x=sample()
        x["grasps"][0,1]=.09
        with self.assertRaisesRegex(ValueError,"identity"):
            check_bank(x)

    def test_failures_confusion(self):
        logits=np.eye(4)
        y=np.array([0,1,2,3])
        result=failure_metrics(logits,y)
        self.assertEqual(result["balanced_accuracy"],1.)
        self.assertEqual(result["macro_f1"],1.)

    def test_native_quality_ranking(self):
        m=quality_stats(np.array([.1,.9,.5]),np.array([0.,1.,.5]))
        self.assertEqual(m["pair_agreement"],1.)
        self.assertEqual(m["top10_quality"],.5)

    def test_reference_geometry_is_diagnostic_only(self):
        x=sample()
        model=EvidenceProbe(32,"reference_geometry","quality")
        with self.assertRaisesRegex(ValueError,"Privileged"):
            model(x)


class CaptureTests(unittest.TestCase):
    def test_six_role_token_pool_and_mask(self):
        class FakeReader(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg=SimpleNamespace(variant="volume_rel",reader_dim=64)
                self.attn=nn.MultiheadAttention(64,4,batch_first=True)
        reader=FakeReader()
        rt=SimpleNamespace(model=SimpleNamespace(kview_grasp_module=
              SimpleNamespace(decoder=SimpleNamespace(reader=reader))))
        capture=GVARCapture(rt)
        tokens=torch.randn(3,36,64)
        weights=torch.zeros((3,1,36))
        weights[:,:,0]=1.
        pad=torch.ones((3,36),dtype=torch.bool)
        pad[:,0]=False
        capture._attn_hook(reader.attn,(torch.zeros(3,1,64),tokens,tokens),
                           dict(key_padding_mask=pad),
                           (torch.zeros(3,1,64),weights))
        self.assertEqual(capture.role_chunks[0].shape,(3,6,64))
        np.testing.assert_allclose(capture.role_chunks[0][:,0,:],
                                   tokens[:,0,:].numpy().astype(np.float16),atol=2e-3)
        self.assertEqual(capture.mass_chunks[0].shape,(3,6))

    def test_capture_requires_strict_reader_variant(self):
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg=SimpleNamespace(variant="slot",reader_dim=64)
                self.attn=nn.MultiheadAttention(64,4,batch_first=True)
        rt=SimpleNamespace(model=SimpleNamespace(kview_grasp_module=
            SimpleNamespace(decoder=SimpleNamespace(reader=Dummy()))))
        with self.assertRaisesRegex(ValueError,"volume_rel"):
            GVARCapture(rt)


class SamplingTests(unittest.TestCase):
    def test_same_native_action_budget_all_seeds(self):
        x=sample(128)
        row=dict(domain="original",scene=3,anno=50)
        a=_frame_subset(x,row)
        b=_frame_subset(x,row)
        np.testing.assert_array_equal(a["physical_id"],b["physical_id"])
        self.assertEqual(len(a["query_id"]),64)
        self.assertEqual(int(a["source_new"].sum()),32)
        self.assertEqual(int((a["source_new"]==0).sum()),32)

    def test_support_frame_selection_fixed_first_last(self):
        entries=[]
        for d in ("original","gntrans"):
            for s in range(100):
                for a in [0,50,100,200]:
                    entries.append(dict(row=dict(domain=d,scene=s,anno=a,split="train")))
        selected=_choose_frames({"entries":entries})
        self.assertEqual(len(selected),400)
        self.assertEqual({e["row"]["anno"] for e in selected},{0,200})
        entries[0]["row"]["scene"]=120
        with self.assertRaises(ValueError):
            _choose_frames({"entries":entries})

    def test_original_old_failure_flags_are_recovered_from_signed_full_cache(self):
        # The intermediate U∪D trace strips collision/empty, so recover only
        # through the two hashed original four-depth P1 source files.
        n=8
        g=np.zeros((n,17),dtype=np.float32)
        g[:,0]=np.array([.4,.7,.95,.5,.5,.6,.75,.4])
        g[:,1]=.06;g[:,2]=.02;g[:,3]=np.tile([.01,.02,.03,.04],2)
        g[:,4:13]=np.eye(3).reshape(9)
        g[:,13]=np.repeat([.0,.1],4);g[:,15]=.6;g[:,16]=-1
        f=np.repeat([0,1],4);d=np.tile(np.arange(4),2)
        ids=physical_ids(g)
        counts=np.array([0,0,0,0,0,0,0,0],np.int64)
        bits=np.zeros((n,6),np.uint8)
        collision=np.array([0,0,1,0,0,0,0,0],np.uint8)
        empty=np.array([0,0,0,0,0,0,1,0],np.uint8)
        pack=dict(family=f,depth=d,grasps=g,physical_id=ids,count=counts,
                  bits=bits,native_sensor_valid=np.array([True,True]))
        with tempfile.TemporaryDirectory() as td:
            arms={}
            for name in ("uniform","decision"):
                path=Path(td)/(name+".npz")
                np.savez_compressed(path,family=f,depth=d,grasps=g,
                    physical_id=ids,count=counts,bits=bits,
                    collision=collision,empty=empty)
                arms[name]=dict(file=str(path),sha256=sha256(path))
            result=_old_native(pack,arms)
            self.assertEqual(result["q"].tolist(),[0,1])
            self.assertEqual(result["collision"].tolist(),[1,0])
            self.assertEqual(result["empty"].tolist(),[0,1])

    def test_native_depth_inference_preserves_geometry(self):
        from unittest.mock import MagicMock
        Q,D=1024,4
        bank=np.zeros((Q,D,17),dtype=np.float32)
        bank[...,0]=np.arange(1,5)[None,:]*.1
        bank[...,1]=.06;bank[...,2]=.02
        bank[...,3]=np.arange(1,5)[None,:]*.01
        bank[...,4:13]=np.eye(3).reshape(9)
        bank[...,13]=np.arange(Q)[:,None]*.0001
        bank[...,15]=.7;bank[...,16]=-1
        snap=dict(grasps=bank,
                  features=np.zeros((Q,D,64),dtype=np.float32),
                  base_logit=np.log(bank[...,0]/(1-bank[...,0])),
                  angles=np.zeros(Q,dtype=int))
        model=EvidenceProbe(32,"score64","quality",width=32)
        class FakeRT:
            def snapshot(self,row,seed):return snap
        fake_capture=MagicMock()
        fake_capture.__enter__.return_value=fake_capture
        fake_capture.chosen.return_value=dict(
            pre_reader=np.zeros((Q,32),dtype=np.float32),
            post_reader=np.zeros((Q,32),dtype=np.float32),
            reader_delta=np.zeros((Q,32),dtype=np.float32),
            role_tokens=np.zeros((Q,6,64),dtype=np.float32),
            role_mass=np.zeros((Q,6),dtype=np.float32),
            role_valid=np.zeros((Q,6),dtype=np.float32),
            pred_geometry=np.zeros((Q,6,6),dtype=np.float32))
        with patch("diagnostics.evidence_decodability.infer.GVARCapture",
                   return_value=fake_capture):
            result,_=native_actions(FakeRT(),dict(scene=0),7,model=model)
        self.assertEqual(result.shape,(Q,17))
        np.testing.assert_array_equal(result[:,1:16],bank[:,-1,1:16])

    def test_high_score_failure_inversion(self):
        x=sample(12)
        x["quality_count"][:]=0
        x["quality_count"][1]=6
        x["nms_role"][:]=2
        x["native_score_percentile"][:]=.9
        old=np.ones(12)*.3;new=old.copy()
        old[0]=.2;old[1]=.8
        new[0]=.95;new[1]=.7
        m=_inversion_counts(old,new,x)
        self.assertGreater(m["newly_wrong"],0)


class CLITests(unittest.TestCase):
    def test_help_all_stages(self):
        for module in ("cache","p0_match","train","audit","infer","compare"):
            cmd=[sys.executable,"-m",f"diagnostics.evidence_decodability.{module}","--help"]
            r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=35)
            self.assertEqual(r.returncode,0,f"{module}: {r.stderr}")


if __name__=="__main__":
    unittest.main()
