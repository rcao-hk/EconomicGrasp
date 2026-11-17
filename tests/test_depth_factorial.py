"""Pure CPU regression tests for depth Oracle A and physical distance × crop B.

No Blender, CUDA, actual user CAD or training data is needed by these tests.
The synthetic planar depth source checks the mathematical protocol only;
it does NOT validate Blender's Cycles depth/seg renderer parity.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from diagnostics.depth_factorial.oracle_core import (
    MODES,spatial_fold_masks,fit_coefficients,evaluate_frame,
    paired_scene_bootstrap,local_gradient_mae
)
from diagnostics.depth_factorial.factorial_contract import (
    PLAN_SCHEMA,SOURCE_SCHEMA,INFER_SCHEMA,
    validate_plan,expected_c2w,verify_fixed_physics,
    source_image,digest,sha256,write_json
)
from diagnostics.depth_factorial.factorial_analyze import run as analyze_factorial


def mock_plan(w=128,h=128):
    return dict(schema=PLAN_SCHEMA,
        object=dict(kind="procedural_box",dimensions_m=[.12,.08,.002]),
        target_world=[0.,0.,0.],
        camera=dict(width=w,height=h,fx=140.,fy=140.,
                    cx=w/2.,cy=h/2.,
                    R_c2w=np.eye(3,dtype=int).tolist()),
        distances_m=[.35,.50,.70],
        render=dict(engine="CYCLES",samples=32,seed=20261117))


class OracleTests(unittest.TestCase):
    def setUp(self):
        y,x=np.indices((96,112))
        self.p=(.33+.15*x/112+.08*y/96).astype(np.float64)
        self.ref=(1.7*self.p+.04+.001*np.sin(x*.04)).astype(np.float64)
        self.mask=np.ones_like(self.p,dtype=bool)

    def test_four_spatial_folds_no_leakage(self):
        total=np.zeros_like(self.mask,dtype=int)
        for fold in range(4):
            train,test=spatial_fold_masks(self.p.shape,fold,guard=8)
            self.assertFalse((train&test).any())
            self.assertLess(int(train.sum()),int((~test).sum()))
            total+=test.astype(int)
        np.testing.assert_array_equal(total,np.ones_like(total))

    def test_affine_recovers_shape_only_on_independent_pixels(self):
        rows=evaluate_frame(self.p,self.ref,self.mask,guard=8,
                            min_train=100,min_test=100,
                            affine_min_std=.001)
        self.assertEqual(len(rows),16)
        for fold in range(4):
            old=next(r for r in rows if r["fold"]==fold and r["mode"]=="identity")
            affine=next(r for r in rows if r["fold"]==fold and r["mode"]=="affine")
            self.assertTrue(affine["valid"])
            self.assertLess(affine["mae"],old["mae"])
            self.assertAlmostEqual(affine["fitted_a"],1.7,delta=.05)
            self.assertAlmostEqual(affine["fitted_b"],.04,delta=.025)

    def test_constant_prediction_affine_fail_closed(self):
        p=np.full(self.p.shape,.42)
        rows=evaluate_frame(p,self.ref,self.mask,guard=8,
                            min_train=100,min_test=100)
        affine=[r for r in rows if r["mode"]=="affine"]
        self.assertTrue(all(not r["valid"] and "variance" in r["reason"]
                            for r in affine))
        self.assertTrue(all(r["valid"] for r in rows if r["mode"]!="affine"))

    def test_negative_affine_is_nonphysical_not_hidden(self):
        train=np.ones_like(self.mask)
        coeff=fit_coefficients("affine",self.p,1.-self.p,train)
        self.assertLess(coeff["a"],0)
        self.assertTrue(coeff["nonphysical"])

    def test_scene_bootstrap_folds_do_not_become_IID(self):
        rows=[]
        for s in (1,2,3):
            for f in (0,1):
                for q in range(4):
                    rows.append(dict(scene=s,frame=f,fold=q,valid=True,
                                     mae=.1*s))
        result=paired_scene_bootstrap(rows,"mae",draws=50,seed=9)
        self.assertEqual(result["scenes"],3)
        self.assertAlmostEqual(result["mean"],.2)


class PhysicalSourceTests(unittest.TestCase):
    def test_camera_distance_truly_changes_translation_only(self):
        plan=mock_plan()
        validate_plan(plan)
        t0=expected_c2w(plan,.35)
        t1=expected_c2w(plan,.70)
        np.testing.assert_allclose(t0[:3,:3],t1[:3,:3])
        np.testing.assert_allclose(t1[:3,3]-t0[:3,3],[0,0,-.35])

    def test_digital_zoom_cannot_be_accepted_as_real_focal_change(self):
        plan=mock_plan()
        wrong=copy.deepcopy(plan)
        wrong["camera"]["fy"]=120.
        with self.assertRaisesRegex(ValueError,"centered fx=fy"):
            validate_plan(wrong)
        wrong=copy.deepcopy(plan)
        wrong["distances_m"]=[.35,.50,.50]
        with self.assertRaisesRegex(ValueError,"unique"):
            validate_plan(wrong)

    def test_trusted_mesh_requires_real_file_hash(self):
        plan=mock_plan()
        plan["object"]=dict(kind="mesh",path="/missing/model.obj",
                            sha256="0"*64,metres_per_mesh_unit=.001)
        with self.assertRaisesRegex(ValueError,"mesh SHA"):
            validate_plan(plan)

    def generate_synthetic_render_and_predictions(self,root):
        """For protocol math: a static opaque planar object, real C2W moves."""
        plan=mock_plan()
        out=Path(root)
        outputs=[]
        pred_recs=[]
        w=plan["camera"]["width"];h=plan["camera"]["height"]
        yy,xx=np.indices((h,w))
        for i,d in enumerate(plan["distances_m"]):
            # Project the same world z=0 rectangular plane at a camera center
            # translated along +Z optical direction. Near projects larger.
            dx=(xx+.5-w/2.)*d/plan["camera"]["fx"]
            dy=(yy+.5-h/2.)*d/plan["camera"]["fy"]
            mask=(np.abs(dx)<=.06)&(np.abs(dy)<=.04)
            image=np.zeros((h,w,3),dtype=np.uint8)
            image[:]=[150,153,157]
            image[mask]=[170,77,45]
            depth=np.where(mask,float(d),0.).astype(np.float32)
            folder=out/f"distance_{i:03d}"
            folder.mkdir(parents=True)
            rgb=folder/"rgb.png";z=folder/"depth_m.npy";seg=folder/"seg.png"
            Image.fromarray(image).save(rgb)
            np.save(z,depth,allow_pickle=False)
            Image.fromarray(np.where(mask,255,0).astype(np.uint8)).save(seg)
            files={name:dict(path=str(path),sha256=sha256(path))
                   for name,path in (("rgb",rgb),("depth",z),("seg",seg))}
            view=dict(distance_m=d,index=i,
                      K=[[140.,0.,w/2.],[0.,140.,h/2.],[0.,0.,1.]],
                      T_c2w=expected_c2w(plan,d).tolist(),files=files)
            outputs.append(view)
            for policy in ("fixed_full","fixed_roi","adaptive_workspace"):
                if policy=="fixed_full":box=(0,0,w,h)
                elif policy=="fixed_roi":box=(30,40,98,88)
                else:
                    sy,sx=np.nonzero(mask)
                    box=(int(sx.min()),int(sy.min()),int(sx.max()+1),int(sy.max()+1))
                # Trained model in this synthetic test outputs constant0.41;
                # shape is still the real 448x448 depth decoder output.
                pred=np.full((448,448),.41,dtype=np.float32)
                path=out/f"pred_{policy}_{i}.npy"
                np.save(path,pred,allow_pickle=False)
                sx0,sy0,sx1,sy1=box
                Kn=[[140.*448/(sx1-sx0),0.,(w/2.-sx0)*448/(sx1-sx0)],
                    [0.,140.*448/(sy1-sy0),(h/2.-sy0)*448/(sy1-sy0)],
                    [0.,0.,1.]]
                pred_recs.append(dict(distance_id=i,distance_m=d,
                    crop_policy=policy,crop_box=list(box),K=Kn,
                    prediction_file=str(path),prediction_sha256=sha256(path),
                    render_source_sha256=files))
        source=dict(schema=SOURCE_SCHEMA,complete=True,plan=plan,
                    plan_sha256=digest(plan),views=outputs)
        srcfile=out/"render_manifest.json";write_json(srcfile,source)
        inferred=dict(schema=INFER_SCHEMA,complete=True,
                      source_manifest_sha256=sha256(srcfile),
                      source_plan_sha256=digest(plan),
                      base_checkpoint_sha256="f"*64,
                      crop_policies=["fixed_full","fixed_roi","adaptive_workspace"],
                      records=pred_recs)
        inferred["manifest_sha256"]=digest(inferred)
        predfile=out/"pred_manifest.json";write_json(predfile,inferred)
        return srcfile,predfile

    def test_signed_physical_sources_are_fixed_camera_and_unmodified(self):
        with tempfile.TemporaryDirectory() as td:
            src,pred=self.generate_synthetic_render_and_predictions(td)
            mf=json.loads(src.read_text())
            verified=verify_fixed_physics(mf)
            self.assertTrue(verified["physical_translation_only"])
            view=mf["views"][0]
            rgb,depth,mask=source_image(view)
            self.assertEqual(depth.shape,(128,128))
            self.assertGreater(mask.sum(),300)
            fake=copy.deepcopy(mf)
            fake["views"][2]["T_c2w"][2][3]+=.2
            with self.assertRaisesRegex(ValueError,"Unexpected camera"):
                verify_fixed_physics(fake)

    def test_same_surface_intersection_across_three_crops(self):
        with tempfile.TemporaryDirectory() as td:
            src,pred=self.generate_synthetic_render_and_predictions(td)
            output=Path(td)/"analysis"
            result=analyze_factorial(type("Config",(),dict(
                 source_manifest=str(src),frozen_manifest=str(pred),
                 output=str(output),min_motion=.03,
                 max_points=128,min_shared=16,abs_tol=.006,
                 rel_tol=.008,xyz_tol=.012))())
            self.assertEqual(len(result["rows"]),6)
            self.assertGreater(result["n_shared_pairs"],30)
            for row in result["rows"]:
                self.assertAlmostEqual(row["slope"],0.,places=6)
                self.assertAlmostEqual(row["crop_distance_interaction_did"],0.,places=6)


if __name__=="__main__":
    unittest.main()
