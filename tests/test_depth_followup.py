"""CPU synthetic tests for P0 crop-distance and P1 same-surface GC6D tracks.

These verify math and fail-closed contracts, NOT real CUDA/GPU, full78 source
compatibility, physical camera calibration or actual CAD grasp quality.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from diagnostics.depth_followup.common import (
    _validate_crop,_frame_meta,
)
from diagnostics.depth_followup.multiview import (
    check_transform,unproject,project,camera_to_world,world_to_camera,
    to_crop,bilinear,sample_anchors,match_pair,verify_c2w_scene_poses,
)
from diagnostics.depth_followup.p0_analyze import (
    scene_bootstrap,binned,_outside_train,
)
from diagnostics.depth_followup.p1_analyze import (
    slope_origin,_summarize_scene,
)
from diagnostics.frozen_depth_response.geometry import crop_K_from_raw


class SameSurfaceGeometry(unittest.TestCase):
    def synthetic_pair(self,z_source=.8,z_target=.6,source_prediction=.4,
                       target_prediction=.4,occluded=False,wrong_label=False):
        h=w=128
        K=np.array([[100.,0.,64.],[0.,100.,64.],[0.,0.,1.]],dtype=float)
        T0=np.eye(4)
        T1=np.eye(4);T1[2,3]=z_source-z_target
        z1=z_target-.2 if occluded else z_target
        s=dict(scene=1,img_id=2,anno=0,
               depth_raw_m=np.full((h,w),z_source,dtype=np.float32),
               seg_raw=np.ones((h,w),dtype=np.int32),
               raw_K=K,K=(K*np.array([[3.5,1,3.5],[1,3.5,3.5],[1,1,1.]])).astype(np.float32),
               T_c2w=T0,crop_box=np.array([0,0,w,h]),
               pred_depth=np.full((448,448),source_prediction,dtype=np.float32))
        t=dict(s)
        t.update(img_id=6,anno=1,
                 depth_raw_m=np.full((h,w),z1,dtype=np.float32),
                 seg_raw=np.full((h,w),2 if wrong_label else 1,dtype=np.int32),
                 T_c2w=T1,
                 pred_depth=np.full((448,448),target_prediction,dtype=np.float32))
        return s,t

    def test_rigid_unprojection_across_camera_depth(self):
        s,t=self.synthetic_pair()
        x=unproject([64],[64],[.8],s["raw_K"])
        xw=camera_to_world(x,s["T_c2w"])
        xc=world_to_camera(xw,t["T_c2w"])
        u,v,z=project(xc,t["raw_K"])
        self.assertAlmostEqual(float(z[0]),.6)
        self.assertAlmostEqual(float(u[0]),64.)
        self.assertAlmostEqual(float(v[0]),64.)
        np.testing.assert_allclose(world_to_camera(
            camera_to_world(x,s["T_c2w"]),s["T_c2w"]),x,atol=1e-10)

    def test_same_point_constant_predictions_expose_zero_distance_slope(self):
        s,t=self.synthetic_pair()
        records,audit=match_pair(s,t,np.array([64]),np.array([64]),np.array([1]))
        self.assertEqual(len(records),1)
        self.assertEqual(audit["matched_pred"],1)
        r=records[0]
        self.assertAlmostEqual(r["delta_reference_m"],-.2,places=6)
        self.assertAlmostEqual(r["delta_predicted_m"],0,places=6)
        self.assertAlmostEqual(slope_origin([r["delta_reference_m"]]*12,
            [r["delta_predicted_m"]]*12,min_samples=10),0.,places=6)

    def test_response_is_unity_only_when_same_surface_tracks_depth(self):
        s,t=self.synthetic_pair(target_prediction=.2)
        records,_=match_pair(s,t,np.array([64]),np.array([64]),np.array([1]))
        self.assertEqual(len(records),1)
        self.assertAlmostEqual(records[0]["delta_predicted_m"],
                               records[0]["delta_reference_m"],places=6)

    def test_occlusion_or_segmentation_change_rejects_false_match(self):
        s,t=self.synthetic_pair(occluded=True)
        rows,audit=match_pair(s,t,[64],[64],[1])
        self.assertEqual(rows,[])
        self.assertEqual(audit["depth_consistent"],0)
        s,t=self.synthetic_pair(wrong_label=True)
        rows,audit=match_pair(s,t,[64],[64],[1])
        self.assertEqual(rows,[])
        self.assertEqual(audit["same_seg"],0)

    def test_crop_projection_uses_original_intrinsics(self):
        raw=np.array([[800.,0.,600.],[0.,850.,400.],[0.,0.,1.]])
        box=(200,100,1000,700)
        new,_=crop_K_from_raw(raw,box,(448,448))
        _validate_crop(new,raw,box,720,1280)
        point=unproject([600],[400],[.7],raw)
        u,v,z=project(point,new)
        u2,v2=to_crop([600],[400],box)
        self.assertAlmostEqual(float(u[0]),float(u2[0]),places=4)
        self.assertAlmostEqual(float(v[0]),float(v2[0]),places=4)
        zeros=np.full((448,448),.42)
        self.assertAlmostEqual(bilinear(zeros,u,v)[0],.42)

    def test_segment_balanced_anchor_sampling_is_reproducible(self):
        depth=np.full((128,128),.8,dtype=np.float32)
        seg=np.ones((128,128),dtype=np.int32)
        seg[:30,:30]=2
        a=sample_anchors(depth,seg,scene=20,img_id=3,max_points=100,
                         points_per_object=35)
        b=sample_anchors(depth,seg,scene=20,img_id=3,max_points=100,
                         points_per_object=35)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertEqual(len(a[0]),70)
        self.assertEqual(set(a[2].tolist()),{1,2})

    def test_scene_gt_C2W_composition_and_invalid_inverse(self):
        s,t=self.synthetic_pair()
        obj_world=np.eye(4)
        obj_world[:3,3]=[0.,0.,1.]
        frames=[s,t]
        scene_gt={}
        for frame in frames:
            T=np.linalg.inv(frame["T_c2w"])@obj_world
            scene_gt[str(frame["img_id"])]=[dict(obj_id=1,
                cam_R_m2c=T[:3,:3].reshape(-1).tolist(),
                cam_t_m2c=(1000*T[:3,3]).tolist())]
        stats=verify_c2w_scene_poses(frames,scene_gt)
        self.assertLess(stats["median_translation_m"],1e-8)
        bad=[dict(s),dict(t)]
        bad[1]["T_c2w"]=np.linalg.inv(t["T_c2w"])
        with self.assertRaisesRegex(RuntimeError,"C2W"):
            verify_c2w_scene_poses(bad,scene_gt)

    def test_nonrigid_transform_refused(self):
        T=np.eye(4);T[:3,:3]=np.diag([2.,1.,1.])
        with self.assertRaisesRegex(ValueError,"Non-rigid"):
            check_transform(T)


class P0Stats(unittest.TestCase):
    def test_crop_metrics_do_not_infer_object_size(self):
        train=[dict(scene=i,median_reference=.4+i*.01,
                    fx_crop=300+i,crop_fraction_w=.6,
                    crop_fraction_h=.7,foreground_fraction=.2)
               for i in range(10)]
        q={k:np.quantile([r[k] for r in train],[.05,.95])
             for k in ("median_reference","fx_crop","fy_crop","crop_fraction_w",
                       "crop_fraction_h","foreground_fraction")
             if k!="fy_crop"}
        q["fy_crop"]=[250.,900.]
        novel=[dict(train[0],scene=11,median_reference=.8,fx_crop=550.)]
        result=_outside_train(novel,q)
        self.assertEqual(result["out_of_original_p05_p95"]["median_reference"],1.)
        self.assertEqual(result["out_of_original_p05_p95"]["fx_crop"],1.)

    def test_scene_bootstrap_uses_scene_aggregation(self):
        rows=[dict(scene=0,mae=.1),dict(scene=0,mae=.2),
              dict(scene=1,mae=.3)]
        result=scene_bootstrap(rows,"mae",draws=50,seed=1)
        self.assertAlmostEqual(result["mean"],.225)
        self.assertEqual(result["scenes"],2)


class TrackStats(unittest.TestCase):
    def test_scene_level_distance_response(self):
        rows=[]
        for s in range(4):
            for j in range(50):
                ref=-.04-.001*j
                rows.append(dict(scene=s,src_img_id=s,src_u=j,src_v=9,
                    dst_img_id=10,delta_reference_m=ref,
                    delta_predicted_m=.25*ref,
                    ref_pose_distance_m=.12,true_depth_dst_m=.7,
                    pred_depth_dst_m=.4,source_error_m=-.3,target_error_m=-.3))
        stats=_summarize_scene(rows,min_motion=.03,min_scene_pairs=40)
        self.assertTrue(stats["usable"])
        self.assertAlmostEqual(stats["slope"],.25)
        self.assertEqual(stats["pairs"],200)


if __name__=="__main__":
    unittest.main()
