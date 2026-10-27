"""CPU-only unit tests for P0-A/D protocol, paths, and fixed-list oracle.

Run: python -m unittest discover -s tests -p 'test_mixed_p0_abcd.py' -v
P0-B/C physical data and P0-D official API integration require server datasets.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.mixed_p0_a_rank_stats import (
    ap_at_k, precision_at_k, load_official, load_candidate_oracle
)
from tools.mixed_p0_d_gc6d_cva_eval import (
    select_indices, identities, check_prediction
)


class TestPairedOfficialAP(unittest.TestCase):
    def test_ap_prefix_is_not_precision_at_rank(self):
        a=np.zeros((30,26,50,6),np.float32)
        a[:,:,0,:]=1.0
        a[:,:,1,:]=.5
        a[:,:,2,:]=1/3
        self.assertAlmostEqual(float(ap_at_k(a,2).mean()),.75)
        self.assertAlmostEqual(float(precision_at_k(a,2).mean()),.5)
        self.assertAlmostEqual(float(ap_at_k(a,3).mean()),(1+.5+1/3)/3,places=6)

    def test_shape_and_range_check(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"accuracy.npy"
            good=np.zeros((30,26,50,6),np.float32)
            np.save(path,good)
            self.assertEqual(load_official(path).shape,(30,26,50,6))
            np.save(path,good[0])
            with self.assertRaises(ValueError):
                load_official(path)
            good[0,0,0,0]=np.nan
            np.save(path,good)
            with self.assertRaises(ValueError):
                load_official(path)

    def test_full_pipeline_exact_scene_paired_delta(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            files=[]
            for run,val in (("point",.2),("gvar",.4)):
                for domain in ("graspnet","gntrans"):
                    for split in ("test_seen","test_similar","test_novel"):
                        path=root/run/domain/"test_collision_off"/"official"/split/"accuracy.npy"
                        path.parent.mkdir(parents=True,exist_ok=True)
                        np.save(path,np.full((30,26,50,6),val,np.float32))
                        files.append(path)
            out=root/"out"
            script=Path(__file__).resolve().parents[1]/"tools/mixed_p0_a_rank_stats.py"
            subprocess.check_call([sys.executable,str(script),
                "--run",f"point={root/'point'}",
                "--run",f"gvar={root/'gvar'}",
                "--output-dir",str(out),
                "--bootstrap","100","--ranks","1,5,10,50"])
            summary=json.loads((out/"summary.json").read_text())
            report=next(x for x in summary["results"] if x["domain"]=="macro" and x["rank_k"]==50)
            self.assertAlmostEqual(report["delta_pp"],20.0,places=4)
            self.assertAlmostEqual(report["ci95_pp"][0],20.0,places=4)
            self.assertAlmostEqual(report["ci95_pp"][1],20.0,places=4)
            self.assertEqual(len(summary["source_sha256"]),12)

    def test_oracle_remains_separate_from_official_ap(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"candidate.csv"
            with p.open("w",newline="") as fd:
                w=csv.DictWriter(fd,fieldnames=["run","domain","split","scene","frame","rank","friction"])
                w.writeheader()
                for rank,fr in enumerate((-1.0,0.2,0.6),1):
                    w.writerow(dict(run="point",domain="graspnet",split="test_seen",
                         scene="scene_0100",frame="0000",rank=rank,friction=fr))
            rows=load_candidate_oracle(p,{"point"},(3,))
            vals={r["method"]:r["diagnostic_ap_percent"] for r in rows}
            self.assertGreater(vals["privileged_oracle"],vals["original"])


class _FakeGC6D:
    camera="realsense-d435"
    samples=[(1,3+4*i,i) for i in range(13)]+[(2,3+4*i,i) for i in range(13)]
    def __len__(self):
        return len(self.samples)


class TestGC6DEvalProtocol(unittest.TestCase):
    def test_sample_indices_are_per_scene(self):
        full=select_indices(_FakeGC6D(),SimpleNamespace(sample_fraction=1.0,max_infer_frames=0))
        self.assertEqual(len(full),26)
        tiny=select_indices(_FakeGC6D(),SimpleNamespace(sample_fraction=0.1,max_infer_frames=0))
        self.assertEqual(tiny,[0,10,13,23])

    def test_official_six_digit_dump_names(self):
        with tempfile.TemporaryDirectory() as d:
            paths=identities(_FakeGC6D(),[0,13],Path(d))
            self.assertEqual(paths[0][2],Path(d)/"000001"/"realsense-d435"/"000003.npy")
            self.assertEqual(paths[1][2],Path(d)/"000002"/"realsense-d435"/"000003.npy")
            f=paths[0][2]
            f.parent.mkdir(parents=True)
            np.save(f,np.zeros((1,17),np.float32))
            self.assertEqual(check_prediction(f),1)
            np.save(f,np.ones((1,16),np.float32))
            with self.assertRaises(ValueError):
                check_prediction(f)


if __name__=="__main__":
    unittest.main()
