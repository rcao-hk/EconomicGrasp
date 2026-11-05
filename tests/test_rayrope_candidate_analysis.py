import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from diagnostics.analyze_rayrope_candidates import (
    load_candidates,
    sample_metrics,
    width_bins,
    matched_width_sweeps,
    candidate_failure_breakdown,
)
from diagnostics.analyze_rayrope_depth_error import depth_metrics, paired_candidate_effect
from diagnostics.sample_rayrope_candidates_uniform import main as sampler_main


MUS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)


def make_rows():
    rows = []
    for sample, scene in (("scene0100_frame0000", 100), ("scene0100_frame0001", 100),
                          ("scene0130_frame0000", 130)):
        # Two widths for one identical action family, plus one collision case.
        specs = [
            ("w04", "family0", 0.04, 0.90, 0.2, False, False),
            ("w06", "family0", 0.06, 0.70, 0.8, False, False),
            ("coll", "family1", 0.05, 0.99, -1.0, True, False),
        ]
        for cid, parent, width, score, friction, collision, empty in specs:
            row = {
                "sample_id": sample, "scene_id": str(scene), "candidate_id": cid,
                "parent_id": parent, "intervention": "native", "score": str(score),
                "width_m": str(width), "insertion_depth_m": "0.025",
                "center_x": "0.1", "center_y": "0.0", "center_z": "0.5",
                "collision_or_empty": str(int(collision)), "empty": str(int(empty)),
                "friction": str(friction),
            }
            for i in range(3):
                for j in range(3):
                    row[f"r{i}{j}"] = str(1.0 if i == j else 0.0)
            for mu in MUS:
                row[f"success_mu_{mu:.1f}"] = str(int(friction > 0 and friction <= mu))
            rows.append(row)
        # Paired reader condition: same physical candidate IDs, changed scores.
        for row in rows[-3:]:
            other = dict(row)
            other["intervention"] = "reader_plus10mm"
            other["score"] = str(float(row["score"]) + (0.05 if row["candidate_id"] == "w04" else -0.02))
            rows.append(other)
    return rows


class RayRoPECandidateAnalysisTests(unittest.TestCase):
    def test_failure_friction_width_and_pairing(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidates.csv"
            rows = make_rows()
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            parsed, schema = load_candidates(str(path), None, "auto")
            self.assertEqual(schema["width_unit_resolved"], "m")
            self.assertEqual(len(parsed), len(rows))
            self.assertEqual(sum(x["failure"] == "collision" for x in parsed), 6)
            self.assertTrue(all(len([x for x in parsed if x["mu_flags"][mu] is not None]) == len(parsed)
                                for mu in MUS))
            native = [x for x in parsed if x["condition"] == "native"]
            grouped = {}
            for x in native:
                grouped.setdefault(x["sample"], []).append(x)
            m = sample_metrics(next(iter(grouped.values())), 0.95)
            self.assertGreaterEqual(m["top1_quality"], 0.0)
            self.assertEqual(m["mu_0.2"]["all_candidate_success_rate"], 1.0 / 3.0)
            bins = width_bins(parsed, [0.0, 0.05, 0.07, float("inf")], "native")
            self.assertEqual(sum(x["n"] for x in bins["bins"]), len(native))
            sweep = matched_width_sweeps(parsed, "native")
            self.assertEqual(sweep["num_matched_width_families"], 3)
            failure = candidate_failure_breakdown(native, 0.95)
            self.assertEqual(failure["all_candidates"]["n"], len(native))

    def test_depth_metrics_and_paired_effect(self):
        pred = np.asarray([[0.50, 0.51], [0.49, 0.52]])
        gt = np.asarray([[0.49, 0.50], [0.50, 0.50]])
        m = depth_metrics(pred, gt, None, "m")
        self.assertAlmostEqual(m["mean_signed_error_m"], 0.0075, places=6)
        self.assertAlmostEqual(m["mae_m"], 0.0125, places=6)
        rows = []
        for cid, s0, s1, q in (("a", 0.9, 0.7, 1.0), ("b", 0.5, 0.8, 0.0)):
            for condition, score in (("native", s0), ("reader_plus10mm", s1)):
                rows.append({"sample": "scene0100_frame0000", "candidate": cid,
                             "condition": condition, "score": score, "quality": q})
        effect = paired_candidate_effect(rows, "native", "reader_plus10mm")
        self.assertEqual(effect["scene0100_frame0000"]["top1_changed"], True)
        self.assertEqual(effect["scene0100_frame0000"]["top1_quality_change"], -1.0)

    def test_depth_error_script_requires_registered_gt_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pred = np.full((4, 4), 0.51, dtype=np.float32)
            gt = np.full((4, 4), 0.50, dtype=np.float32)
            np.save(root / "pred.npy", pred)
            np.save(root / "gt.npy", gt)
            manifest = root / "depth_manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["sample_id", "pred_path", "gt_path"])
                w.writeheader()
                w.writerow({"sample_id": "scene0100_frame0000",
                            "pred_path": str(root / "pred.npy"),
                            "gt_path": str(root / "gt.npy")})
            self.assertTrue(manifest.is_file())

    def test_uniform_sampler_requires_full_pool_acknowledgement(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "pool.csv"
            with src.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["sample_id", "candidate_id", "score"])
                w.writeheader()
                for i in range(20):
                    w.writerow({"sample_id": "s0", "candidate_id": str(i), "score": str(100-i)})
            out = root / "sampled.csv"
            cmd = [sys.executable, "diagnostics/sample_rayrope_candidates_uniform.py",
                   "--input_csv", str(src), "--output_csv", str(out), "--n_per_sample", "5"]
            denied = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(denied.returncode, 0)
            cmd.append("--input_is_full_pool")
            ok = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, msg=ok.stderr)
            with out.open(newline="", encoding="utf-8") as f:
                sampled = list(csv.DictReader(f))
            self.assertEqual(len(sampled), 5)
            self.assertEqual(len({x["candidate_id"] for x in sampled}), 5)


if __name__ == "__main__":
    unittest.main()
