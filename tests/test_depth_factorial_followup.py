"""CPU synthetic regressions for follow-up P0 constant/physical-affine + P1 data.

No real GC6D, Blender, CUDA, CAD grader or backbone training is involved.
The tests verify same-fold fit isolation, oracle physical constraints,
synthetic RGB saturation refusal and CAD train/test split contracts.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from diagnostics.depth_factorial.oracle_core import (
    evaluate_frame,spatial_fold_masks
)
from diagnostics.depth_factorial.oracle_followup_p0 import (
    ARMS,evaluate_all,fit_followup,edge_pair_metrics,mean_centered_shape
)
from diagnostics.depth_factorial.scale_cues_p1 import rgb_quality
from diagnostics.depth_factorial.distance_split_p1 import (
    register,read_registered
)
from diagnostics.depth_factorial.factorial_contract import (
    PLAN_SCHEMA,SOURCE_SCHEMA,sha256,digest,write_json,expected_c2w,
    verify_fixed_physics
)


class ConstantAndAffineOracleTests(unittest.TestCase):
    def setUp(self):
        y,x=np.indices((96,112))
        self.pred=(.35+.11*x/112+.04*y/96).astype(float)
        self.ref=(1.4*self.pred+.08+
                  .002*np.sin(x/7)).astype(float)
        self.fg=np.ones_like(self.pred,dtype=bool)

    def test_exact_original_oracle_on_same_folds_and_pixels(self):
        old=evaluate_frame(self.pred,self.ref,self.fg,
                           guard=8,min_train=128,min_test=64,
                           affine_min_std=.003)
        new=evaluate_all(self.pred,self.ref,self.fg,
                         guard=8,min_train=128,min_test=64,
                         affine_min_std=.003)
        self.assertEqual(len(new),4*len(ARMS))
        for r in old:
            v=next(k for k in new if k["fold"]==r["fold"] and
                   k["mode"]==r["mode"])
            self.assertEqual(r["valid"],v["valid"])
            if r["valid"]:
                self.assertEqual(r["fit_pixels"],v["fit_pixels"])
                self.assertEqual(r["test_pixels"],v["test_pixels"])
                self.assertAlmostEqual(r["mae"],v["mae"],places=12)
                self.assertAlmostEqual(r["gradient_mae"],v["gradient_mae"],places=12)

    def test_constant_fit_only_uses_nonheldout_region(self):
        fit,heldout=spatial_fold_masks(self.pred.shape,fold=2,guard=8)
        a=fit_followup("constant_mean",self.pred,self.ref,fit)
        b=fit_followup("constant_median",self.pred,self.ref,fit)
        changed=self.ref.copy()
        changed[heldout]+=100
        c=fit_followup("constant_mean",self.pred,changed,fit)
        d=fit_followup("constant_median",self.pred,changed,fit)
        self.assertAlmostEqual(a["b"],c["b"],places=12)
        self.assertAlmostEqual(b["b"],d["b"],places=12)
        self.assertEqual(a["a"],0)
        self.assertEqual(b["a"],0)

    def test_negative_affine_is_clamped_to_reference_mean(self):
        ref=1.4-self.pred
        fit=np.ones_like(self.fg,dtype=bool)
        raw=fit_followup("affine",self.pred,ref,fit)
        constrained=fit_followup("affine_nonnegative",self.pred,ref,fit)
        constant=fit_followup("constant_mean",self.pred,ref,fit)
        self.assertLess(raw["a"],0)
        self.assertTrue(constrained["clamped"])
        self.assertEqual(constrained["a"],0)
        self.assertAlmostEqual(constrained["b"],constant["b"],places=12)

    def test_positively_correlated_affine_equals_constrained(self):
        fit=np.ones_like(self.fg,dtype=bool)
        raw=fit_followup("affine",self.pred,self.ref,fit)
        positive=fit_followup("affine_nonnegative",self.pred,self.ref,fit)
        self.assertGreater(positive["a"],0)
        self.assertFalse(positive["clamped"])
        self.assertAlmostEqual(raw["a"],positive["a"],places=12)
        self.assertAlmostEqual(raw["b"],positive["b"],places=12)

    def test_near_constant_affine_refuses_to_invent_shape(self):
        pred=np.full_like(self.pred,.410)
        fit=np.ones_like(self.fg,dtype=bool)
        restricted=fit_followup("affine_nonnegative",pred,self.ref,fit,min_std=.003)
        self.assertFalse(restricted["valid"])
        self.assertIn("variance",restricted["reason"])
        constant=fit_followup("constant_mean",pred,self.ref,fit)
        self.assertTrue(constant["valid"])

    def test_reference_edge_gradient_classification(self):
        ref=np.full((64,64),.55)
        ref[:,32:]=.80
        good=edge_pair_metrics(ref,ref,np.ones_like(ref,dtype=bool))
        const=edge_pair_metrics(np.full_like(ref,.65),ref,
                                np.ones_like(ref,dtype=bool))
        self.assertGreater(good["edge_pairs"],0)
        self.assertAlmostEqual(good["edge_grad_mae"],0)
        self.assertGreater(const["edge_grad_mae"],.2)
        self.assertGreater(const["smooth_pairs"],0)
        self.assertAlmostEqual(const["smooth_grad_mae"],0)
        self.assertLess(mean_centered_shape(np.full_like(ref,.65),
            ref,np.ones_like(ref,dtype=bool)),.5)

    def test_all_modes_on_identical_holdout_support(self):
        new=evaluate_all(self.pred,self.ref,self.fg,guard=8,
                         min_train=128,min_test=64)
        for fold in range(4):
            vals=[r for r in new if r["fold"]==fold]
            self.assertEqual(len({r["test_pixels"] for r in vals}),1)
            self.assertEqual(len({r["fit_pixels"] for r in vals}),1)
            self.assertTrue(all(r["valid"] for r in vals))


class CorrectedPhysicalSourceTests(unittest.TestCase):
    def _make_source(self,root,dims):
        objroot=Path(root)/("obj_"+str(round(dims[0]*1000)))
        objroot.mkdir(parents=True)
        plan=dict(schema=PLAN_SCHEMA,
            object=dict(kind="procedural_box",dimensions_m=list(dims)),
            target_world=[0.,0.,0.],
            camera=dict(width=128,height=128,fx=140.,fy=140.,
                        cx=64.,cy=64.,
                        R_c2w=np.eye(3).tolist()),
            distances_m=[.35,.45,.55,.70,.85],
            render=dict(engine="CYCLES",samples=32,seed=20261117))
        entries=[]
        for i,d in enumerate(plan["distances_m"]):
            path=objroot/f"distance_{i:03d}"
            path.mkdir()
            rgb=np.full((128,128,3),[65,70,80],dtype=np.uint8)
            rgb[45:85,43:83]=[165,75,42]
            mask=np.zeros((128,128),dtype=np.uint8)
            mask[45:85,43:83]=255
            z=np.where(mask>0,d,0.).astype(np.float32)
            files={}
            for key,p in (("rgb",path/"rgb.png"),
                          ("seg",path/"seg.png"),
                          ("depth",path/"depth_m.npy")):
                if key=="rgb":Image.fromarray(rgb).save(p)
                elif key=="seg":Image.fromarray(mask).save(p)
                else:np.save(p,z,allow_pickle=False)
                files[key]=dict(path=str(p),sha256=sha256(p))
            entries.append(dict(index=i,distance_m=d,
                K=[[140.,0.,64.],[0.,140.,64.],[0.,0.,1.]],
                T_c2w=expected_c2w(plan,d).tolist(),files=files))
        source=dict(schema=SOURCE_SCHEMA,plan=plan,
                    plan_sha256=digest(plan),complete=True,views=entries)
        p=objroot/"manifest.json"
        write_json(p,source)
        return dict(manifest=str(p),sha256=sha256(p))

    def test_reject_near_white_original_render(self):
        white=np.full((64,64,3),255,dtype=np.uint8)
        fg=np.ones((64,64),dtype=bool);fg[:10]=False
        with self.assertRaises(RuntimeError):
            rgb_quality(white,fg)
        good=np.full((64,64,3),[60,70,80],dtype=np.uint8)
        good[fg]=[175,77,48]
        q=rgb_quality(good,fg)
        self.assertGreater(q["rgb_fg_bg_contrast"],5.)
        self.assertLess(q["rgb_white_fraction"],.96)

    def test_preregistered_synthetic_split_and_object_overlap_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=self._make_source(root,(.02,.03,.04))
            b=self._make_source(root,(.03,.03,.04))
            c=self._make_source(root,(.04,.03,.04))
            train=root/"train.json";eval=root/"eval.json"
            train.write_text(json.dumps([a,b]))
            eval.write_text(json.dumps([c]))
            review=root/"review.json"
            review.write_text(json.dumps(dict(documented=False,
                warning="procedural synthetic CPU smoke no CAD identity claim")))
            args=SimpleNamespace(train_sources=str(train),eval_sources=str(eval),
                 object_overlap_review=str(review),smoke=True,
                 output=str(root/"registered.json"))
            result=register(args)
            self.assertEqual(len(result["train"]),2)
            self.assertEqual(len(read_registered(args.output)["eval"]),1)
            eval.write_text(json.dumps([a]))
            args.output=str(root/"bad.json")
            with self.assertRaisesRegex(ValueError,"overlap"):
                register(args)

    def test_unchanged_optical_camera_intrinsics(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=self._make_source(tmp,(.02,.03,.04))
            source=json.loads(Path(raw["manifest"]).read_text())
            info=verify_fixed_physics(source)
            self.assertTrue(info["fixed_intrinsics"])
            source["views"][1]["K"][0][0]+=20.
            with self.assertRaisesRegex(ValueError,"K changed"):
                verify_fixed_physics(source)


if __name__=="__main__":
    unittest.main()
