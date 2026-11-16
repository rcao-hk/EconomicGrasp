"""Pure CPU tests for GC6D frozen depth camera/crop/distance diagnostics.

No actual GC6D images, source checkpoint, CUDA or model forward here.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from diagnostics.frozen_depth_response.geometry import (
    crop_K_from_raw,recover_crop_box,pixel_affine,update_K,
    warp_tensor,check_projective_consistency,paired_depth_metrics,
    matched_warp_response,
)
from diagnostics.frozen_depth_response.analyze import (
    rank_average,spearman,slope,summarize,
)
from diagnostics.frozen_depth_response.run import _scene_list


class CameraGeometryTests(unittest.TestCase):
    def test_native_gc6d_crop_intrinsics_recovery(self):
        raw=np.array([[910.,0.,960.],[0.,925.,540.],[0.,0.,1.]])
        bbox=(256,120,1500,930)
        cropped,_=crop_K_from_raw(raw,bbox,(448,448))
        got,err=recover_crop_box(raw,cropped.astype(np.float32),
                                 (1080,1920),(448,448))
        self.assertEqual(got,bbox)
        self.assertLess(err,2e-3)

    def test_crop_failure_not_silently_accepted(self):
        raw=np.array([[900.,0.,960.],[0.,900.,540.],[0.,0.,1.]])
        cropped,_=crop_K_from_raw(raw,(200,100,1720,980),(448,448))
        cropped[0,2]+=1.4
        with self.assertRaises((RuntimeError,ValueError)):
            recover_crop_box(raw,cropped,(1080,1920),(448,448))

    def test_projective_consistency_all_pre_registered_variants(self):
        K=np.array([[570.,0.,224.],[0.,850.,224.],[0.,0.,1.]])
        for name in ("identity","zoom_in_115","zoom_out_085",
                     "shift_right_08","shift_down_08"):
            A=pixel_affine(name)
            err=check_projective_consistency(K,A)
            self.assertLess(err,3e-4)
            Kn=update_K(K,A)
            self.assertAlmostEqual(float(Kn[0,0]),570.*A[0,0],places=3)

    def test_zoom_in_does_not_change_same_ray_depth_on_ideal_warp(self):
        h=w=64
        xx=np.arange(w)[None,:]
        yy=np.arange(h)[:,None]
        base=(.55+xx*.003+yy*.001).astype(np.float32)
        gt=(base+0.03).astype(np.float32)
        fg=np.ones((h,w),bool)
        A=pixel_affine("zoom_in_115",(h,w))
        pred,_=warp_tensor(torch.from_numpy(base)[None,None],A)
        pred=pred[0,0].numpy()
        metric=matched_warp_response(base,pred,gt,fg,A,min_pixels=100)
        self.assertGreater(metric["common_ray_n"],100)
        self.assertLess(abs(metric["same_ray_median_delta"]),1e-7)
        self.assertLess(abs(metric["same_ray_median_ratio"]-1),1e-6)
        self.assertGreater(metric["overlap_fraction"],.5)

    def test_zoom_out_mask_excludes_synthetic_padded_pixels(self):
        base=np.full((64,64),.6,np.float32)
        gt=np.full((64,64),.7,np.float32)
        fg=np.ones((64,64),bool)
        A=pixel_affine("zoom_out_085",(64,64))
        pred,_=warp_tensor(torch.from_numpy(base)[None,None],A)
        m=matched_warp_response(base,pred[0,0].numpy(),gt,fg,A,min_pixels=10)
        self.assertLess(m["overlap_fraction"],1.)
        self.assertGreater(m["overlap_fraction"],.5)

    def test_reference_only_on_valid_foreground(self):
        pred=np.array([[.4,.4],[.4,.4]])
        ref=np.array([[.7,0],[.8,.7]])
        fg=np.array([[True,True],[False,True]])
        d=paired_depth_metrics(pred,ref,fg,min_pixels=2)
        self.assertTrue(d["valid"])
        self.assertEqual(d["valid_pixels"],2)
        self.assertAlmostEqual(d["median_error"],-.3)
        e=paired_depth_metrics(pred,ref,fg,min_pixels=3)
        self.assertFalse(e["valid"])


class DistanceStatsTests(unittest.TestCase):
    def test_rank_with_ties(self):
        np.testing.assert_allclose(rank_average([1,2,2,4]),[1,2.5,2.5,4])
        self.assertAlmostEqual(spearman([1,2,3],[2,4,6]),1.)
        self.assertAlmostEqual(spearman([1,2,3],[6,4,2]),-1.)

    def test_distance_slope_is_not_scalar_scale_calibration(self):
        x=np.array([.4,.5,.7,.9])
        y=.3+.2*x
        self.assertAlmostEqual(slope(x,y,min_span=.03),.2)
        self.assertIsNone(slope([.5,.51,.515],[.5,.6,.7],min_span=.03))

    def test_scene_aggregator_uses_scene_unit_and_intervention_id(self):
        rows=[]
        for scene in (11,12,13):
            for anno in range(13):
                ref=.5+anno*.025+scene*.001
                pred=.3+ref*.2
                base=dict(scene=scene,img_id=anno,anno=anno,
                          intervention="identity",
                          sensor_median_pred=pred,
                          sensor_median_reference=ref,
                          sensor_median_ratio=pred/ref,
                          sensor_mae=abs(pred-ref),
                          sensor_median_error=pred-ref,
                          same_ray_median_delta=0.,
                          same_ray_median_abs_delta=0.,
                          same_ray_median_ratio=1.,pose_delta_deg=0.,
                          K_only_maxabs=0.,overlap_fraction=1.)
                rows.append(base)
                if anno==0:
                    rows.append({**base,"intervention":"K_only_negative_control",
                                 "K_only_maxabs":0.,"same_ray_median_delta":0.})
                    rows.append({**base,"intervention":"zoom_in_115",
                                 "same_ray_median_delta":.03,
                                 "same_ray_median_abs_delta":.03,
                                 "same_ray_median_ratio":1.08})
        m=dict(status="complete",expected_frames=39,
               output_csv_sha256="dummy")
        result=summarize(rows,m,bootstrap=100,seed=7)
        self.assertEqual(result["scenes"],3)
        self.assertEqual(result["scenes_with_valid_slope"],3)
        self.assertAlmostEqual(result["within_scene_slope"]["mean"],.2)
        self.assertAlmostEqual(
            result["interventions"]["zoom_in_115"]["same_ray_median_abs_delta"]["mean"],.03)

    def test_scene_list_is_explicit_sorted_unique(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"scenes.json"
            path.write_text(json.dumps([4,1,3]))
            self.assertEqual(_scene_list(path),[1,3,4])
            path.write_text(json.dumps([1,1]))
            with self.assertRaises(ValueError):_scene_list(path)
            path.write_text(json.dumps([1,"2"]))
            with self.assertRaises(ValueError):_scene_list(path)


if __name__=="__main__":
    unittest.main()
