"""CPU contract regressions for same-ray gradient and 200/600 exposure study.

Actual network forward/Blender/CAD physical source/117 pretrained DPT is NOT
part of these CPU tests; see preflight, gradient and budget scripts for GPU
source acceptance. No synthetic test results are presented as real AP.
"""
from __future__ import annotations

from collections import Counter
import unittest

import numpy as np

from diagnostics.depth_factorial.gradient_conflict_core import (
    CROPS,gradient_pair,shared_raw_points,per_group_vectors,
    deterministic_seed,summarize_cases
)
from diagnostics.depth_factorial.distance_budget_core import (
    budget_schedule,exposure_summary,assert_600_pair
)
from diagnostics.depth_factorial.distance_crosscrop_compare import _ci
from diagnostics.depth_factorial.factorial_contract import (
    PLAN_SCHEMA,validate_plan
)


class BudgetScheduleTests(unittest.TestCase):
    def test_600_total_steps_and_exposure_are_exact(self):
        roi=budget_schedule("roi",600,objects=2)
        multi=budget_schedule("multi3",600,objects=2)
        verified=assert_600_pair(roi,multi)
        self.assertTrue(verified["total_step_matched"])
        self.assertTrue(verified["per_crop_roi200_vs_multi600"])
        self.assertEqual(Counter(r["crop_policy"] for r in roi),{"fixed_roi":600})
        self.assertEqual(Counter(r["crop_policy"] for r in multi),
                         {c:200 for c in CROPS})
        self.assertEqual(Counter(r["distance_id"] for r in multi),
                         {0:200,2:200,4:200})
        nine=verified["multi_nine_cell_counts"]
        self.assertEqual(len(nine),9)
        self.assertEqual(sum(nine.values()),600)
        self.assertEqual(set(nine.values()),{66,67})
        self.assertEqual(exposure_summary(multi)["supervised_pixels"],153600)

    def test_six_hundred_arms_have_same_object_and_distance_step_order(self):
        roi=budget_schedule("roi",600,objects=2)
        multi=budget_schedule("multi3",600,objects=2)
        self.assertEqual(
            [(r["object_index"],r["distance_id"]) for r in roi],
            [(r["object_index"],r["distance_id"]) for r in multi])
        self.assertEqual(
            [r["object_index"] for r in roi[:200]],
            [r["object_index"] for r in budget_schedule("roi",200,objects=2)])
        self.assertEqual(
            [(r["distance_id"],r["crop_policy"]) for r in roi[:200]],
            [(r["distance_id"],r["crop_policy"]) for r in budget_schedule("roi",200,objects=2)])
        self.assertEqual(
            [(r["distance_id"],r["crop_policy"]) for r in multi[:200]],
            [(r["distance_id"],r["crop_policy"]) for r in budget_schedule("multi3",200,objects=2)])

    def test_different_object_sequence_or_distance_is_rejected(self):
        roi=budget_schedule("roi",600,objects=2)
        multi=budget_schedule("multi3",600,objects=2)
        wrong=[dict(r) for r in multi]
        wrong[2]["object_index"]=1-wrong[2]["object_index"]
        with self.assertRaisesRegex(ValueError,"object or distance"):
            assert_600_pair(roi,wrong)
        wrong=[dict(r) for r in multi]
        wrong[3]["distance_id"]=1
        with self.assertRaisesRegex(ValueError,"object or distance"):
            assert_600_pair(roi,wrong)


class GradientDiagnosticTests(unittest.TestCase):
    def test_cosine_interference_is_signed_and_normalized(self):
        g=np.array([1.,0.,.5])
        with_neg=np.array([-1.,0.,-.5])
        r=gradient_pair(g,with_neg)
        self.assertAlmostEqual(r["cosine"],-1.,places=12)
        self.assertTrue(r["first_order_conflict"])
        self.assertAlmostEqual(r["source_step_transfer"],-1.,places=12)
        with_pos=gradient_pair(g,g)
        self.assertAlmostEqual(with_pos["cosine"],1.)
        self.assertFalse(with_pos["first_order_conflict"])
        zero=gradient_pair(g,np.zeros_like(g))
        self.assertIsNone(zero["cosine"])
        self.assertIsNone(zero["first_order_conflict"])

    def test_gradient_grouping_keeps_order_and_identity(self):
        grp=per_group_vectors(
            ["refine.a","refine.b","output.weight"],
            [np.array([1,2]),np.array([3,4]),np.array([5])])
        np.testing.assert_array_equal(grp["refine"],[1,2,3,4])
        np.testing.assert_array_equal(grp["output"],[5])

    def test_same_physical_world_ray_is_selected_independently_of_crop(self):
        mask=np.ones((96,96),bool)
        depth=np.ones((96,96),np.float32)*.70
        boxes=dict(fixed_full=(0,0,96,96),
                   fixed_roi=(16,12,80,84),
                   adaptive_workspace=(22,22,74,76))
        seed=deterministic_seed("sample_mesh_SHA",2)
        u,v,z=shared_raw_points(mask,depth,boxes,max_points=128,
                                 margin=3,seed=seed)
        u2,v2,z2=shared_raw_points(mask,depth,boxes,max_points=128,
                                   margin=3,seed=seed)
        np.testing.assert_array_equal(u,u2)
        np.testing.assert_array_equal(v,v2)
        np.testing.assert_array_equal(z,z2)
        self.assertEqual(len(u),128)
        self.assertTrue(np.all((u>=25)&(u<71)&(v>=25)&(v<73)))
        changed=dict(boxes,adaptive_workspace=(0,0,12,12))
        with self.assertRaisesRegex(ValueError,"common physical rays"):
            shared_raw_points(mask,depth,changed,max_points=128,seed=seed)

    def test_cosine_aggregate_counts_cases_not_pixels(self):
        rows=[dict(state="base",crop_a="fixed_full",crop_b="fixed_roi",
                   layer="all",cosine=c,distance_id=d)
              for c,d in ((-.3,0),(.6,2),(.4,4))]
        agg=summarize_cases(rows)
        self.assertEqual(agg["base/fixed_full/fixed_roi"]["cases"],3)
        self.assertAlmostEqual(
            agg["base/fixed_full/fixed_roi"]["negative_fraction"],1/3)


class RendererMaterialContractTests(unittest.TestCase):
    def base_plan(self):
        return dict(schema=PLAN_SCHEMA,
            target_world=[0.,0.,0.],
            object=dict(kind="procedural_box",
                        dimensions_m=[.05,.06,.07]),
            camera=dict(width=128,height=128,fx=140.,fy=140.,
                cx=64.,cy=64.,R_c2w=np.eye(3).tolist()),
            distances_m=[.35,.45,.55,.70,.85],
            render=dict(engine="CYCLES",samples=32,seed=20261117))

    def test_per_CAD_material_can_change_but_is_source_explicit(self):
        plan=self.base_plan()
        plan["appearance"]=dict(base_color_linear=[.13,.22,.50],
                                roughness=.25)
        self.assertEqual(validate_plan(plan),plan)
        invalid=self.base_plan()
        invalid["appearance"]=dict(base_color_linear=[2.,.22,.5],
                                   roughness=.25)
        with self.assertRaisesRegex(ValueError,"material"):
            validate_plan(invalid)
        invalid=self.base_plan()
        invalid["appearance"]=dict(base_color_linear=[.5,.5,.5],
                                   roughness=1.2)
        with self.assertRaisesRegex(ValueError,"material"):
            validate_plan(invalid)

    def test_small_CAD_bootstrap_does_not_invent_population_precision(self):
        x=_ci([.03,-.06],draws=50,seed=9)
        self.assertEqual(x["n_cad"],2)
        self.assertAlmostEqual(x["mean"],-.015)
        self.assertIsNone(_ci([.2],draws=50,seed=9)["ci95"])


if __name__=="__main__":
    unittest.main()
