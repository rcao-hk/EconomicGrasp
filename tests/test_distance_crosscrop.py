"""Pure CPU regression tests for P0 same-ray audit and P1 orthogonal crop×distance.

These DO NOT run the CUDA frozen DPT head or use real CAD files. Real 117
execution must additionally check every signed source, geometry and checkpoint.
"""
from __future__ import annotations

from collections import Counter
import unittest

import numpy as np

from diagnostics.depth_factorial.distance_crosscrop_core import (
    schedule,verify_four_arm_protocols,CROPS,MAIN_STEPS,MAIN_PIXELS,MAIN_SEED,
)
from diagnostics.depth_factorial.crop_consistency_core import (
    sample_crop_predictions,evaluate_shared_points,
)
from diagnostics.depth_factorial.distance_crosscrop_compare import _ci


class OrthogonalFactorialTests(unittest.TestCase):
    def test_wide_multi3_sees_all_nine_cells_approximately_equally(self):
        seq=schedule(200,MAIN_SEED,objects=2,
                     arm="near_mid_far",crop_design="multi3")
        counts=Counter((r["distance_id"],r["crop_policy"]) for r in seq)
        self.assertEqual(len(counts),9)
        self.assertEqual(sum(counts.values()),200)
        self.assertLessEqual(max(counts.values())-min(counts.values()),1)
        self.assertEqual([seq[i]["distance_id"] for i in range(9)],
                         [0,2,4]*3)
        self.assertEqual([seq[i]["crop_policy"] for i in range(9)],
            [CROPS[0],CROPS[1],CROPS[2],
             CROPS[1],CROPS[2],CROPS[0],
             CROPS[2],CROPS[0],CROPS[1]])

    def test_near_multi3_has_same_crop_schedule_as_wide_multi3(self):
        near=schedule(200,MAIN_SEED,2,"near_only","multi3")
        wide=schedule(200,MAIN_SEED,2,"near_mid_far","multi3")
        self.assertEqual([r["object_index"] for r in near],
                         [r["object_index"] for r in wide])
        self.assertEqual([r["crop_policy"] for r in near],
                         [r["crop_policy"] for r in wide])
        self.assertEqual({r["distance_id"] for r in near},{0})
        self.assertEqual({r["distance_id"] for r in wide},{0,2,4})
        self.assertEqual(Counter(r["crop_policy"] for r in near),
                         Counter({"fixed_full":67,"fixed_roi":67,
                                  "adaptive_workspace":66}))

    def _protocols(self):
        seq=schedule(200,MAIN_SEED,2,"near_only","roi")
        objects=["CAD_A" if r["object_index"]==0 else "CAD_B"
                 for r in seq]
        common=dict(base_checkpoint_sha256="a"*64,
            initial_depth_head_sha256="b"*64,
            registration_identity_sha256="c"*64,
            train_registration_sha256="d"*64,
            steps=MAIN_STEPS,pixels_per_step=MAIN_PIXELS,
            total_supervised_pixels=MAIN_STEPS*MAIN_PIXELS,
            learning_rate=2e-5,weight_decay=1e-4,
            smooth_l1_beta=.03,gradient_clip=1.,seed=MAIN_SEED,
            optimizer="AdamW",trainable_module="depth_net.depthnet.depth_head ONLY",
            trainable_params=10890305,train_object_sequence=objects)
        protocols={}
        for label,arm,crop in (
            ("roi_near","near_only","roi"),("roi_wide","near_mid_far","roi"),
            ("multi_near","near_only","multi3"),
            ("multi_wide","near_mid_far","multi3")):
            events=schedule(MAIN_STEPS,MAIN_SEED,2,arm,crop)
            protocols[label]=dict(common,arm=arm,
                crop_policy="fixed_roi" if crop=="roi" else "multi3",
                train_distance_indices=[r["distance_id"] for r in events])
            if crop=="multi3":
                protocols[label]["train_crop_indices"]=[
                    r["crop_policy"] for r in events]
        return protocols

    def test_legacy_and_new_four_arm_contract_passes(self):
        p=self._protocols()
        out=verify_four_arm_protocols(p,"d"*64)
        self.assertTrue(out["verified"])
        self.assertEqual(out["steps"],200)
        self.assertEqual(out["pixels"],256)

    def test_confounded_crop_schedule_is_rejected(self):
        p=self._protocols()
        # This is the tempting but INVALID choice: distance and crop both
        # rotate every step, so only 3 of the 9 cells get supervised.
        p["multi_wide"]["train_crop_indices"]=[
            CROPS[i%3] for i in range(200)]
        with self.assertRaisesRegex(ValueError,"schedule"):
            verify_four_arm_protocols(p,"d"*64)

    def test_extra_training_budget_and_seed_are_rejected(self):
        p=self._protocols()
        p["multi_wide"]["total_supervised_pixels"]+=1
        with self.assertRaisesRegex(ValueError,"total_supervised_pixels"):
            verify_four_arm_protocols(p,"d"*64)
        p=self._protocols()
        p["multi_near"]["seed"]+=1
        with self.assertRaisesRegex(ValueError,"seed"):
            verify_four_arm_protocols(p,"d"*64)


class SameRayCropTests(unittest.TestCase):
    def synthetic(self):
        # All three 448 crops contain the same RAW (u,v) world-pixel set.
        u=np.arange(60,72)
        v=np.arange(61,73)
        reference=np.full(len(u),.70,dtype=np.float64)
        images={
            "fixed_full":dict(pred=np.full((448,448),.40,dtype=np.float32),
                              crop_box=(0,0,128,128)),
            "fixed_roi":dict(pred=np.full((448,448),.42,dtype=np.float32),
                             crop_box=(40,40,92,92)),
            "adaptive_workspace":dict(pred=np.full((448,448),.50,dtype=np.float32),
                                     crop_box=(50,50,90,90)),
        }
        return u,v,reference,images

    def test_shared_raw_ray_identity_and_crosscrop_depth_disagreement(self):
        u,v,z,images=self.synthetic()
        outcome=sample_crop_predictions(u,v,z,images,min_points=8)
        self.assertEqual(outcome["points"],12)
        self.assertAlmostEqual(outcome["crosscrop_range"],.10,places=6)
        self.assertAlmostEqual(outcome["abs_full_roi"],.02,places=6)
        self.assertAlmostEqual(outcome["abs_adaptive_roi"],.08,places=6)
        rows=evaluate_shared_points(outcome)
        values={r["crop_policy"]:r for r in rows}
        self.assertAlmostEqual(values["fixed_full"]["mae"],.30,places=6)
        self.assertAlmostEqual(values["fixed_roi"]["mae"],.28,places=6)
        self.assertAlmostEqual(values["adaptive_workspace"]["mae"],.20,places=6)
        self.assertEqual(len({v["points"] for v in rows}),1)

    def test_disjoint_crop_cannot_silently_change_sample(self):
        u,v,z,images=self.synthetic()
        images["adaptive_workspace"]["crop_box"]=(0,0,20,20)
        with self.assertRaisesRegex(RuntimeError,"Too few"):
            sample_crop_predictions(u,v,z,images,min_points=8)

    def test_missing_crop_refused_and_unchanged_predictions_zero_discrepancy(self):
        u,v,z,images=self.synthetic()
        images.pop("fixed_full")
        with self.assertRaises(ValueError):
            sample_crop_predictions(u,v,z,images)
        u,v,z,images=self.synthetic()
        for x in images.values():
            x["pred"]=np.full((448,448),.4,dtype=np.float32)
        data=sample_crop_predictions(u,v,z,images,min_points=8)
        self.assertAlmostEqual(data["crosscrop_range"],0,places=8)
        self.assertAlmostEqual(data["crosscrop_std"],0,places=8)

    def test_cad_object_bootstrap_does_not_use_pixels_as_iid(self):
        s=_ci([-.02,-.06],draws=50,seed=7)
        self.assertEqual(s["n_cad"],2)
        self.assertAlmostEqual(s["mean"],-.04)
        single=_ci([.2],draws=50,seed=7)
        self.assertIsNone(single["ci95"])


if __name__=="__main__":
    unittest.main()
