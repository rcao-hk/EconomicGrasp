import unittest

import numpy as np

from diagnostics.rayrope_p1_p2 import (
    analyze_candidate_table,
    depth_affine,
    depth_bias,
    depth_local_plane,
)


class RayRoPEP1P2Tests(unittest.TestCase):
    def test_depth_interventions_preserve_invalid_pixels(self):
        depth = np.asarray([[0.0, 0.5], [np.nan, 1.0]], dtype=np.float32)
        shifted = depth_bias(depth, 0.01)
        self.assertEqual(float(shifted[0, 1]), 0.51)
        self.assertEqual(float(shifted[1, 1]), 1.01)
        self.assertEqual(float(shifted[0, 0]), 0.0)
        self.assertTrue(np.isnan(shifted[1, 0]))

    def test_affine_and_local_plane(self):
        depth = np.ones((3, 5), dtype=np.float32)
        scaled = depth_affine(depth, scale=1.1, bias_m=-0.01)
        self.assertTrue(np.allclose(scaled, 1.09))
        plane = depth_local_plane(depth, bias_m=0.01, slope_x=0.02)
        self.assertAlmostEqual(float(plane[1, 0]), 0.99, places=6)
        self.assertAlmostEqual(float(plane[1, -1]), 1.03, places=6)

    def test_candidate_ranking_and_regret(self):
        rows = []
        for intervention, scores in {
            "native": [0.9, 0.6, 0.2],
            "reader_depth_plus10mm": [0.7, 0.8, 0.1],
        }.items():
            for cid, score, quality in zip(("a", "b", "c"), scores, (1.0, 0.0, 0.0)):
                rows.append({"sample_id": "s0", "candidate_id": cid,
                             "intervention": intervention, "score": str(score),
                             "quality": str(quality)})
        result = analyze_candidate_table(rows, success_threshold=0.5)
        self.assertEqual(result["summary"]["native"]["top1_success"], 1.0)
        self.assertEqual(result["summary"]["reader_depth_plus10mm"]["top1_success"], 0.0)
        self.assertEqual(result["summary"]["reader_depth_plus10mm"]["top1_regret"], 1.0)
        self.assertEqual(len(result["paired_intervention_stability"]), 1)


if __name__ == "__main__":
    unittest.main()
