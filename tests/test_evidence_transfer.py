"""CPU synthetic contracts: P0 physical residuals, label replay and P1 gates.

These tests do NOT claim 117's compiled NMS or CAD evaluator was executed.
No actual held-out external dataset is included in this repository.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from diagnostics.decision_supervision.core import BASE_SHA, physical_ids
from diagnostics.evidence_transfer.contract import (
    SCHEMA, digest, sha256, save_npz, save_json, load_manifest,
    validate_frame, failure_classes, geometry_error, pose_attributes,
)
from diagnostics.evidence_transfer.residual import (
    _inversions, per_action, run as residual_run,
)
from diagnostics.evidence_transfer.holdout import (
    PROV_SCHEMA, gate, metrics_for_frame, read_ledger,
)
from diagnostics.evidence_transfer.intervene import intervene, INTERVENTIONS
from diagnostics.evidence_transfer.collect import _frame


def sample(n=6):
    g=np.zeros((n,17),dtype=np.float32)
    g[:,0]=np.linspace(.91,.51,n)
    g[:,1]=.06;g[:,2]=.02;g[:,3]=.02
    g[:,4:13]=np.eye(3,dtype=np.float32).reshape(9)
    g[:,13]=np.arange(n)*.05
    g[:,15]=.6;g[:,16]=-1
    counts=np.array(([6,0,5,0,3,0]*((n+5)//6))[:n],dtype=np.uint8)
    bits=np.array([np.r_[np.zeros(6-int(q),dtype=np.uint8),
                            np.ones(int(q),dtype=np.uint8)] for q in counts])
    coll=np.zeros(n,np.uint8);empty=np.zeros(n,np.uint8)
    if n>1:coll[1]=1
    if n>3:empty[3]=1
    masks={}
    for method in ("native","score64","all"):
        selected=np.zeros(n,np.uint8)
        selected[:min(n,3)]=1
        masks[f"top10_{method}"]=selected.copy()
        masks[f"top50_{method}"]=selected.copy()
    return dict(grasps=g,physical_id=physical_ids(g),
                native_percentile=np.linspace(.25,1.,n).astype(np.float32),
                score_native=np.linspace(.9,.4,n).astype(np.float32),
                score_score64=np.linspace(.5,.4,n).astype(np.float32),
                score_all=np.linspace(.5,.8,n).astype(np.float32),
                quality_bits=bits,
                assigned_obj=np.zeros(n,np.int64),
                pure_collision=coll,empty=empty,**masks)


def ledger_sources(root):
    base=[f"GraspNet::{x}" for x in range(100)]
    dev=[f"GraspNet::{x}" for x in range(100,190)]
    bp=root/"base_scenes.json";dp=root/"development.json"
    bp.write_text(json.dumps(base));dp.write_text(json.dumps(dev))
    x=dict(schema=PROV_SCHEMA,base_checkpoint_sha256=BASE_SHA,
           base_training_scene_keys=base,development_scene_keys=dev,
           base_training_manifest_path=str(bp),
           base_training_manifest_sha256=sha256(bp),
           development_log_path=str(dp),
           development_log_sha256=sha256(dp),
           object_overlap_review=dict(documented=False,disjoint_verified=False))
    x["ledger_sha256"]=digest(x)
    path=root/"ledger.json"
    path.write_text(json.dumps(x))
    return path


class SchemaTests(unittest.TestCase):
    def test_identity_and_failure_contract(self):
        x=sample()
        self.assertEqual(validate_frame(x),6)
        self.assertEqual(failure_classes(x).tolist(),[0,1,0,2,0,3])
        x["grasps"][0,1]=.07
        with self.assertRaisesRegex(ValueError,"Physical-action"):
            validate_frame(x)

    def test_evidence_reference_error_fail_closed(self):
        x=sample()
        self.assertIsNone(geometry_error(x))
        p=np.zeros((6,6,6),dtype=np.float32)
        ref=np.zeros_like(p)
        p[...,4]=1.;ref[...,4]=1.
        ref[...,3]=.25
        x["pred_geometry"]=p
        x["reference_geometry"]=ref
        validate_frame(x)
        np.testing.assert_allclose(geometry_error(x),.25)
        self.assertEqual(len(pose_attributes(x["grasps"])["polar_bin"]),6)

    def test_signed_manifest_and_modified_file(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);x=sample()
            path=root/"scene.npz"
            save_npz(path,**x)
            data=dict(schema=SCHEMA,base_checkpoint_sha256=BASE_SHA,complete=True,
                      frames=[dict(dataset="GraspNet",split="test_novel",
                                  scene=160,anno=0,file=str(path),sha256=sha256(path))],
                      provenance=dict(prior_development_use=True))
            data["manifest_sha256"]=digest(data)
            mf=root/"manifest.json";save_json(mf,data)
            self.assertEqual(len(load_manifest(mf)["frames"]),1)
            path.write_bytes(b"tampered")
            from diagnostics.evidence_transfer.contract import load_frame
            with self.assertRaises(ValueError):
                load_frame(data["frames"][0])

    def test_top10_must_be_contained_in_top50(self):
        x=sample()
        x["top50_native"][0]=0
        with self.assertRaisesRegex(ValueError,"Top10"):
            validate_frame(x)


class ResidualTests(unittest.TestCase):
    def test_quality_error_uses_identical_action_labels(self):
        x=sample()
        p=per_action(x,dict(dataset="GraspNet",split="test_novel",scene=160,anno=0))
        self.assertEqual(len(p),6)
        self.assertEqual(p[1]["failure"],1)
        self.assertAlmostEqual(p[0]["delta_score"],
                               float(x["score_all"][0]-x["score_score64"][0]))

    def test_bad_zero_vs_good_pair_detection(self):
        x=sample()
        x["native_percentile"][:]=.9
        x["score_score64"][:]=.2
        x["score_all"][:]=.2
        x["score_score64"][0]=.9
        x["score_score64"][1]=.1
        x["score_all"][0]=.3
        x["score_all"][1]=.99
        inv=_inversions(x)
        self.assertGreaterEqual(inv["new_wrong_pairs"],1)

    def test_residual_report_uses_scenes_not_candidate_iid(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            frames=[]
            for scene in (160,161):
                x=sample()
                path=root/f"{scene}.npz"
                save_npz(path,**x)
                frames.append(dict(dataset="GraspNet",split="test_novel",
                    scene=scene,anno=0,file=str(path),sha256=sha256(path)))
            mf=dict(schema=SCHEMA,base_checkpoint_sha256=BASE_SHA,
                    complete=True,frames=frames,provenance={})
            mf["manifest_sha256"]=digest(mf)
            save_json(root/"manifest.json",mf)
            report=residual_run(SimpleNamespace(manifest=str(root/"manifest.json"),
                output=str(root/"report"),bootstrap=50,min_cell=1,seed=17))
            self.assertEqual(report["scene_conditional"]["GraspNet/test_novel"]["scenes"],2)


class ProvenanceTests(unittest.TestCase):
    def test_history_lower_bound_and_actual_source_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=read_ledger(ledger_sources(Path(td)))
            self.assertEqual(len(ledger["base_training_scene_keys"]),100)
            ledger["development_scene_keys"].remove("GraspNet::188")
            ledger.pop("ledger_sha256")
            ledger["ledger_sha256"]=digest(ledger)
            wrong=Path(td)/"bad.json"
            wrong.write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ValueError,"already inspected"):
                read_ledger(wrong)

    def test_novel_is_not_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=read_ledger(ledger_sources(Path(td)))
            bundle=dict(frames=[dict(dataset="GraspNet",scene=160,
                                      split="test_novel",anno=0)])
            result=gate(bundle,ledger,require_untouched=False)
            self.assertFalse(result["untouched_scene_claim"])
            with self.assertRaisesRegex(ValueError,"Previously trained/inspected"):
                gate(bundle,ledger,require_untouched=True)

    def test_external_provenance_is_only_provisionally_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=read_ledger(ledger_sources(Path(td)))
            bundle=dict(frames=[dict(dataset="ExternalDemo",scene="new001",
                                      split="external_holdout",anno=0)],
                        provenance=dict(prior_development_use=False,
                                        dataset_source_sha256="a"*64,
                                        external_evaluator_name="documented offline quality"))
            result=gate(bundle,ledger,require_untouched=True)
            self.assertTrue(result["untouched_scene_claim"])
            self.assertFalse(result["object_disjoint_proven"])
            self.assertEqual(result["scenes"],1)

    def test_metrics_on_same_matched_quality_actions(self):
        x=sample()
        stats=metrics_for_frame(x)
        self.assertIn("delta_all_score64_top10_quality",stats)
        self.assertIn("delta_all_native_q0_score",stats)


class InterventionTests(unittest.TestCase):
    def test_role_mask_and_shuffle_do_not_change_action_identity(self):
        x=dict(query_id=np.array([0,1,2]),
               depth_id=np.array([1,1,1]),
               grasps=sample(3)["grasps"],
               physical_id=sample(3)["physical_id"],
               quality_count=np.array([6,0,1]),
               quality_bits=np.array([[1]*6,[0]*6,[0,0,0,0,0,1]]),
               role_tokens=np.ones((3,6,64),np.float32),
               role_mass=np.ones((3,6),np.float32),
               role_valid=np.ones((3,6),np.float32),
               pred_geometry=np.arange(108,dtype=np.float32).reshape(3,6,6),
               pre_reader=np.ones((3,64),np.float32),
               post_reader=np.ones((3,64),np.float32),
               reader_delta=np.ones((3,64),np.float32))
        for mode in INTERVENTIONS:
            y=intervene(x,mode,7)
            np.testing.assert_array_equal(y["physical_id"],x["physical_id"])
            np.testing.assert_array_equal(y["grasps"],x["grasps"])
            np.testing.assert_array_equal(y["quality_bits"],x["quality_bits"])
        masked=intervene(x,"mask_role_palm",7)
        self.assertEqual(int(masked["role_tokens"][:,4].sum()),0)
        self.assertEqual(int(masked["role_tokens"][:,0].sum()),3*64)


class CollectorTests(unittest.TestCase):
    def test_compiled_grasp_nms_is_required_for_real_collection(self):
        # A fake compiler can test only bookkeeping; it does NOT establish
        # parity with the server's official compiled graspnetAPI NMS.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            g=sample(5)["grasps"]
            g[:,13]=np.array([0,.01,.1,.2,.3],dtype=np.float32)
            methods={}
            for name in ("native","score64","all"):
                folder=root/name
                folder.mkdir()
                arr=g.copy()
                arr[:,0]=(np.array([.9,.8,.7,.6,.5]) if name!="all"
                           else np.array([.4,.95,.7,.6,.5]))
                np.save(folder/"0000.npy",arr)
                rec=dict(path="0000.npy",sha256=sha256(folder/"0000.npy"))
                methods[name]=(dict(interval=10),folder,{(160,0):rec})
            class Eval:
                def evaluate(self,scene,anno,grasp):
                    n=len(grasp)
                    friction=np.full(n,.6)
                    return SimpleNamespace(assigned_obj=np.zeros(n,dtype=int),
                        pure_collision=np.zeros(n,dtype=bool),
                        empty=np.zeros(n,dtype=bool),friction=friction)
            def nms(arr):
                ids=np.argsort(-arr[:,0]);keep=[]
                for i in ids:
                    if not any(abs(arr[i,13]-arr[j,13])<.03 for j in keep):
                        keep.append(int(i))
                return arr[keep]
            artifact,audit,ap= _frame(methods,(160,0),nms,Eval(),None,
                 "GraspNet","test_novel",SimpleNamespace(allow_relabel=True))
            self.assertEqual(validate_frame(artifact),len(artifact["physical_id"]))
            self.assertTrue(all(a.shape==(50,6) for a in ap.values()))


if __name__=="__main__":
    unittest.main()
