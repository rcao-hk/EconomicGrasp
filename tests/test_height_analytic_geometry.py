"""CPU unit tests for the physical decoder (no GraspNet/Minkowski required)."""
import unittest
import numpy as np
import torch

from height_analytic_geometry import (
    plane_from_camera_to_table, height_from_depth, height_to_depth,
    ray_plane_alignment,
)


class HeightAnalyticGeometryTests(unittest.TestCase):
    def setUp(self):
        self.K = torch.tensor([[[2., 0., 2.], [0., 2., 2.], [0., 0., 1.]]])
        self.plane = torch.tensor([[0., 0., -1., .70]])
        self.raw = torch.zeros(1, 1, 5, 5)

    def test_zero_initialized_height_midpoint(self):
        z, h, valid = height_to_depth(self.raw, self.K, self.plane)
        self.assertTrue(valid.all())
        self.assertTrue(torch.allclose(h, torch.full_like(h, .15)))
        self.assertTrue(torch.allclose(z, torch.full_like(z, .55)))

    def test_known_camera_translation_is_analytic(self):
        z1, _, _ = height_to_depth(self.raw, self.K, self.plane)
        plane2 = self.plane.clone(); plane2[:, 3] += .20
        z2, _, _ = height_to_depth(self.raw, self.K, plane2)
        self.assertTrue(torch.allclose(z2-z1, torch.full_like(z1, .20)))

    def test_decode_roundtrip_to_height(self):
        z, h, valid = height_to_depth(self.raw, self.K, self.plane)
        self.assertTrue(torch.allclose(height_from_depth(z, self.K, self.plane), h, atol=1e-6))

    def test_crop_intrinsics_change_ray_angle_not_geometry(self):
        # Pixel (2,2) in original K corresponds to (4,4) in scaled K.
        k2 = self.K.clone(); k2[:, 0, 0] *= 2; k2[:, 1, 1] *= 2
        k2[:, 0, 2] = 4; k2[:, 1, 2] = 4
        dot1,_ = ray_plane_alignment(self.K,self.plane,(5,5))
        dot2,_ = ray_plane_alignment(k2,self.plane,(9,9))
        self.assertAlmostEqual(float(dot1[0,0,2,2]),float(dot2[0,0,4,4]),places=6)

    def test_grazing_ray_is_not_given_plausible_depth(self):
        plane = torch.tensor([[1., 0., 0., .7]])
        z, _, valid = height_to_depth(self.raw, self.K, plane)
        self.assertFalse(bool(valid[0,0,2,2]))
        self.assertEqual(float(z[0,0,2,2]),0.)

    def test_valid_negative_depth_keeps_gradient(self):
        plane = torch.tensor([[0.,0.,-1.,.10]])
        raw = torch.zeros(1,1,5,5,requires_grad=True)
        z,_,valid=height_to_depth(raw,self.K,plane)
        self.assertTrue(valid.all()); self.assertLess(float(z.mean()),0.)
        (z-.3).square().mean().backward()
        self.assertGreater(float(raw.grad.abs().sum()),0.)

    def test_height_gradients_flow(self):
        raw = self.raw.clone().requires_grad_()
        z,h,_=height_to_depth(raw,self.K,self.plane)
        (z-.4).square().mean().backward()
        self.assertGreater(float(raw.grad.abs().sum()),0.)

    def test_table_pose_and_units(self):
        T = np.eye(4); T[:3,:3] = np.diag([1.,-1.,-1.]); T[2,3]=.70
        p=plane_from_camera_to_table(T)
        np.testing.assert_allclose(p, np.array([0.,0.,-1.,.7]),atol=1e-6)
        T[2,3]=700.0
        with self.assertRaises(ValueError): plane_from_camera_to_table(T)

    def test_reflection_rejected(self):
        T=np.eye(4);T[0,0]=-1;T[2,3]=.7
        with self.assertRaises(ValueError):plane_from_camera_to_table(T)

    def test_corrupted_plane_or_K_fails(self):
        with self.assertRaises(ValueError):
            height_to_depth(self.raw,self.K,torch.tensor([[0.,0.,-2.,.7]]))
        with self.assertRaises(ValueError):
            height_to_depth(self.raw,torch.zeros_like(self.K),self.plane)

    def test_height_range_validation(self):
        with self.assertRaises(ValueError):
            height_to_depth(self.raw,self.K,self.plane,height_min_m=.4,height_max_m=.1)


if __name__=="__main__":
    unittest.main()
