"""CPU tests for checkpoint-free CAD action sweep preparation and P1 pairing."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from diagnostics.analyze_rayrope_candidates import load_candidates
from diagnostics.analyze_rayrope_p1_reader import analyze as analyze_reader
from diagnostics.eval_rayrope_controlled_sweeps_from_csv import _row, make_controlled


class RayRoPEControlledCSVTests(unittest.TestCase):
    def _candidate_csv(self, path):
        cols = ["sample_id","scene_id","anno_id","condition","candidate_id",
                "score","width_m","insertion_depth_m","center_x","center_y","center_z",
                "r00","r01","r02","r10","r11","r12","r20","r21","r22",
                "quality","friction"]
        identity = [1,0,0,0,1,0,0,0,1]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for i in range(4):
                row = {
                    "sample_id": "scene0100_frame0000",
                    "scene_id": "100", "anno_id": "0",
                    "condition":"native",
                    "candidate_id":f"seed{i}",
                    "score":str(.9-i*.1),"width_m":str(.05+i*.01),
                    "insertion_depth_m":".02",
                    "center_x":str(i*.01),"center_y":"0","center_z":".5",
                    "quality":str(i/3),"friction":".6",
                }
                row.update({f"r{k//3}{k%3}":str(v) for k,v in enumerate(identity)})
                writer.writerow(row)

    def test_sweep_fixed_physical_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"candidates.csv"
            self._candidate_csv(path)
            raw,schema=load_candidates(str(path),None,"m")
            recs=make_controlled(raw,widths_m=(.04,.06,.08,.1),
                                 depths_m=(.01,.02,.03,.04),
                                 num_anchors=3,seed=10,
                                 original_condition="native",max_width_m=.1)
            self.assertGreater(len(recs),0)
            groups={}
            for sample,original,cond,parent,grasp,widthmode in recs:
                groups.setdefault((cond,parent),[]).append((original,grasp))
                self.assertEqual(len(grasp),17)
                self.assertTrue(np.isfinite(grasp).all())
            for (cond,parent),items in groups.items():
                original=items[0][0]
                row=_row(original)
                for _,grasp in items:
                    self.assertTrue(np.allclose(grasp[4:16],row[4:16]))
                    if cond=="width_sweep":
                        self.assertAlmostEqual(float(grasp[3]),float(row[3]),places=6)
                        self.assertAlmostEqual(float(grasp[0]),float(row[0]),places=6)
                    elif cond=="insertion_sweep_quality_only":
                        self.assertAlmostEqual(float(grasp[1]),float(row[1]),places=6)
                        self.assertAlmostEqual(float(grasp[0]),float(row[0]),places=6)

    def test_reader_pairs_freeze_quality_and_track_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            csvpath=root/"p1.csv"
            cols=["sample_id","scene_id","split","condition","candidate_id",
                  "score","quality","width_m","insertion_depth_m",
                  "center_x","center_y","center_z"]
            rows=[]
            for candidate,q in (("a",1.0),("b",0.0)):
                for cond,score in (("native",.8 if candidate=="a" else .6),
                                   ("coords_only_p10mm",.7 if candidate=="a" else .9),
                                   ("mask_only_p10mm",.8 if candidate=="a" else .6)):
                    rows.append({
                        "sample_id":"scene0100_frame0000","scene_id":"100","split":"seen",
                        "condition":cond,"candidate_id":candidate,
                        "score":score,"quality":q,"width_m":.06,
                        "insertion_depth_m":.02,"center_x":.01,"center_y":0,"center_z":.5,
                    })
            with csvpath.open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=cols)
                writer.writeheader()
                writer.writerows(rows)
            stats=[
                {"sample":"scene0100_frame0000","condition":"coords_only_p10mm",
                 "mask_flip_count":0,"token_count":98,"queries_with_mask_flip":0,"query_count":2},
                {"sample":"scene0100_frame0000","condition":"mask_only_p10mm",
                 "mask_flip_count":0,"token_count":98,"queries_with_mask_flip":0,"query_count":2},
            ]
            manifest=root/"interventions.json"
            manifest.write_text(json.dumps({"checkpoint_sha256":"synthetic",
                                            "mask_statistics":stats}),encoding="utf-8")
            result=analyze_reader(str(csvpath),str(manifest),n_boot=20,seed=3)
            self.assertEqual(result["num_frames"],1)
            self.assertEqual(result["conditions"]["coords_only_p10mm"]["metrics"]["top1_changed"]["mean"],1.0)
            self.assertEqual(result["conditions"]["coords_only_p10mm"]["metrics"]["top1_quality_change"]["mean"],-1.0)
            self.assertEqual(result["conditions"]["mask_only_p10mm"]["metrics"]["mask_flip_fraction"]["mean"],0.0)


if __name__=="__main__":
    unittest.main()
