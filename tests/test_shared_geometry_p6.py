"""P6 CPU regression tests. No server data or official test access."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
import numpy as np
from shared_geometry.data import sha256,signed,atomic_json,load_signed
from shared_geometry.p0_relative_decision import metrics_for_family,scene_balanced
from shared_geometry.p4_common import write_records
from shared_geometry.p6_opportunity import analyze as p60,load_matched
from shared_geometry.p6_transmission import decompose
from shared_geometry.p6_independent_eval import validate_manifest
from shared_geometry.p6_official_ap import validate_model_args,register,execute,assert_complete_dump


class P6GeometryTests(unittest.TestCase):
    @staticmethod
    def visible():
        x=np.zeros((7,75,6),np.float32)
        x[...,2]=1
        return x

    def test_common_error_does_not_change_action_contrast(self):
        gt=self.visible()
        for k in range(7):gt[k,:,0]=k/20.
        pred=gt.copy();pred[...,0]+=.1
        corr=gt.copy();corr[...,0]+=.05
        r=decompose(pred,corr,gt,.03)
        self.assertEqual(r["all_n_sites"],75)
        self.assertAlmostEqual(r["all_udf_total_improvement_mm2"],6.75,places=4)
        self.assertAlmostEqual(r["all_udf_shared_improvement_mm2"],6.75,places=4)
        self.assertAlmostEqual(r["all_udf_contrast_improvement_mm2"],0.,places=5)
        self.assertAlmostEqual(r["all_udf_action_relative_mae_improvement_mm"],0.,places=5)
        self.assertAlmostEqual(r["all_udf_correction_common_energy_fraction"],1.,places=5)

    def test_action_specific_error_is_visible(self):
        gt=self.visible();pred=gt.copy();pred[1:,:,1]+=.1
        r=decompose(pred,gt,gt,.03)
        self.assertGreater(r["all_projective_contrast_improvement_mm2"],0.)
        self.assertAlmostEqual(r["all_projective_total_improvement_mm2"],
            r["all_projective_shared_improvement_mm2"]+
            r["all_projective_contrast_improvement_mm2"],places=6)

    def test_unknown_is_null_not_zero(self):
        gt=self.visible();pred=gt.copy()
        pred[...,2]=0;pred[...,5]=1
        r=decompose(pred,gt,gt,.03)
        self.assertEqual(r["all_n_sites"],0)
        self.assertIsNone(r["all_udf_total_improvement_mm2"])

    def test_rejects_misaligned_actions(self):
        x=np.zeros((6,75,6),np.float32)
        with self.assertRaises(ValueError):decompose(x,x,x,.03)


class P6OpportunityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.names=("predicted_p2_fixed","corrected_p2_fixed","clean_p2_fixed",
                    "predicted_retrained","corrected_adapted")
        self.rows={}
        for arm in self.names:
            rr=[]
            for sid in (80,81):
                counts=np.array([4,6,3,4,4,4,4])
                logits=np.array([5.,6. if ("corrected" in arm or arm=="clean_p2_fixed") else 4.,
                                 0.,0.,0.,0.,0.])
                r=metrics_for_family(logits,counts,[0,.005,-.005,.01,-.01,.02,-.02])
                r.update(scene=sid,domain="original",anno=8,family=0,native_count=4,
                         physical_ids=[f"{sid}_{i}" for i in range(7)])
                rr.append(r)
            self.rows[arm]=rr
        artifacts={k:write_records(self.root/(k+"_families.jsonl.gz"),v)
                   for k,v in self.rows.items()}
        self.summary=dict(schema="shared-visible-geometry-p5-reader-v1",
            kind="matched-reader-comparison",partition="validation",
            weights_matched_budget=True,training_quality_labels_only=True,
            cross_source_scores_are_OOD=True,official_ap_not_run=True,
            validation_fingerprint="synthetic",sources={"synthetic":True},
            scores={k:{"gain":scene_balanced(v,"gain")} for k,v in self.rows.items()},
            artifacts=artifacts)
        self.file=self.root/"p5.json";atomic_json(self.file,signed(self.summary))

    def tearDown(self):self.tmp.cleanup()

    def test_rescue_opportunity_and_flip(self):
        _,_,keys=load_matched(self.file,smoke=True)
        self.assertEqual(len(keys),2)
        r=p60(self.file,self.root/"p6",bootstrap=20,smoke=True)
        self.assertAlmostEqual(r["modes"]["predicted_p2_fixed"]["oracle_gain"],2/6)
        self.assertAlmostEqual(r["modes"]["corrected_p2_fixed"]["rescue_given_opportunity"],1.)
        self.assertAlmostEqual(r["decision_flips"]["changed_selection"],1.)
        with self.assertRaises(FileExistsError):
            p60(self.file,self.root/"p6",smoke=True)

    def test_sha_mismatch_rejected(self):
        with (self.root/"predicted_p2_fixed_families.jsonl.gz").open("ab") as f:
            f.write(b"x")
        with self.assertRaises(ValueError):load_matched(self.file,smoke=True)

    def test_physical_truth_mismatch_rejected(self):
        self.rows["corrected_p2_fixed"][0]["alternatives"][0]["q"]=0.
        self.summary["artifacts"]["corrected_p2_fixed"]=write_records(
            self.root/"changed.jsonl.gz",self.rows["corrected_p2_fixed"])
        atomic_json(self.root/"modified.json",signed(self.summary))
        with self.assertRaises(ValueError):
            load_matched(self.root/"modified.json",smoke=True)


class P6EvaluationGuards(unittest.TestCase):
    def test_independent_contract_rejects_overlap_and_test(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/"external.npz";src.write_bytes(b"fake")
            m=dict(schema="shared-visible-geometry-p6-external-bank-v1",
                   partition="independent_holdout",external_dataset_id="external-scans",
                   quality_independently_evaluated=True,
                   clean_geometry_never_model_input=True,frozen_model_before_holdout=True,
                   depth_corrector_sha256="x"*64,
                   entries=[dict(row=dict(scene=1,anno=8,domain="original"),
                                 path=str(src),sha256=sha256(src))],
                   training_scene_keys=[["graspnet",0]],
                   development_scene_keys=[["graspnet",80]],
                   official_test_scene_keys=[["graspnet",100]])
            validate_manifest(m,"x"*64,smoke=True)
            m["training_scene_keys"]=[["external-scans",1]]
            with self.assertRaises(ValueError):validate_manifest(m,"x"*64,smoke=True)
            m["training_scene_keys"]=[["graspnet",0]]
            m["external_dataset_id"]="graspnet"
            with self.assertRaises(ValueError):validate_manifest(m,"x"*64,smoke=True)

    def test_official_registration_is_dry_run_and_frozen(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);repo=root/"repo";(repo/"utils").mkdir(parents=True)
            for p in ("eval.py","inference_cva.py","utils/arguments.py"):
                (repo/p).write_text("pass\n")
            ck=root/"deploy.pt";ck.write_bytes(b"fixed-checkpoint")
            data=root/"dataset";data.mkdir()
            reg=root/"registry.json"
            register(reg,ck,data,["--multi_modal","--use_cdf","--kview_mode","A1"],
                     repo_root=repo,attest=True)
            commands=execute(reg,root/"official",run=False)
            self.assertEqual(len(commands),3)
            self.assertEqual(commands[0]["split"],"test_seen")
            self.assertFalse((root/"official").exists())
            with self.assertRaises(ValueError):assert_complete_dump(root/"official","test_seen")
            ck.write_bytes(b"mutated")
            with self.assertRaises(ValueError):execute(reg,root/"official",run=False)

    def test_rejects_privileged_ap_flags(self):
        for flag in ("--use_obs_depth","--use_gt_depth","--test_mode","--sample_interval"):
            with self.assertRaises(ValueError):
                validate_model_args(["--multi_modal",flag,"1"])


if __name__=="__main__":unittest.main()
