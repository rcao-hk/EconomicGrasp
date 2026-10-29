"""CPU-only synthetic contracts for P1-0/P1-1/P1-2 and collision-on protocol.

Run from repo root:
  python -m unittest discover -s tests -p 'test_mixed_p1_gc6d.py' -v
Requires numpy, scipy, torch; no real dataset, CUDA or official GC6D API.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from mixed_p1_gc6d_coverage import raw_object_stats, frame_summary, read_scene_ids
from mixed_p1_gc6d_compare import load_accuracy, per_scene_ap, compare_arrays
from mixed_p1_gc6d_collision_eval import (
    atomic_json, atomic_npy, expected_paths, full_identity_audit,
    ensure_manifest, validate_grasps, parse_args,
)
from mixed_p1_historical import DeterministicIndexedDataset, sample_source_indices
from mixed_p1_source_gradients import dot_grads


class TestRawGC6DLabels(unittest.TestCase):
    def test_mutually_exclusive_failure_causes_and_point_filter(self):
        # P=2; V=A=1; D=4. First point physically graspable at .5,
        # but its only .2 candidate collides, so the <=.4 point filter
        # removes a point that still has >=.5 high-friction successes.
        fric=np.asarray([.2,.5,.7,1.3,.1,.2,0,0],np.float32).reshape(2,1,1,4)
        width=np.full_like(fric,.05)
        width[1,0,0,1]=.15
        collision=np.zeros_like(fric,bool)
        collision[0,0,0,0]=True
        x=raw_object_stats(fric,width,collision,.4,.1)
        self.assertEqual(x["raw_candidates"],8)
        self.assertEqual(x["bad_friction_candidates"],3)
        self.assertEqual(x["bad_width_given_friction"],1)
        self.assertEqual(x["collision_given_friction_width"],1)
        self.assertEqual(x["successful_candidates"],3)
        self.assertEqual(x["kept_points_expected"],1)
        self.assertEqual(x["physically_graspable_points"],2)
        self.assertEqual(x["graspable_points_lost_by_04_filter"],1)

    def test_matches_cdf_later_width_mask_distinction(self):
        fric=np.asarray([.2],np.float32).reshape(1,1,1,1)
        width=np.asarray([.12],np.float32).reshape(1,1,1,1)
        coll=np.zeros_like(fric,bool)
        x=raw_object_stats(fric,width,coll,.4,.1)
        self.assertEqual(x["kept_points_expected"],1)
        self.assertEqual(x["successful_candidates"],0)

    def test_split_strictness(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"scene_ids.json"
            p.write_text(json.dumps(list(range(232))))
            self.assertEqual(len(read_scene_ids(p)),232)
            p.write_text(json.dumps([0]*232))
            with self.assertRaises(ValueError):
                read_scene_ids(p)


class TestMatchedQueryCounts(unittest.TestCase):
    def test_distinguish_matched_and_positive_actions(self):
        valid=torch.zeros((1,2,2,2),dtype=torch.bool)
        bins=torch.zeros_like(valid,dtype=torch.uint8)
        valid[0,0,0,0]=True
        valid[0,1,1,1]=True
        bins[0,0,0,0]=3
        result=frame_summary({
            "batch_grasp_cdf_valid_mask":valid,
            "batch_grasp_cdf_bins_angle_depth":bins,
        })
        self.assertEqual(result["matched_queries"],2)
        self.assertEqual(result["matched_actions"],2)
        self.assertEqual(result["positive_actions"],1)
        self.assertFalse(result["zero_positive_frame"])
        bins.zero_()
        self.assertTrue(frame_summary({
            "batch_grasp_cdf_valid_mask":valid,
            "batch_grasp_cdf_bins_angle_depth":bins,
        })["zero_positive_frame"])


class TestPairedGC6DAP(unittest.TestCase):
    def test_exact_paired_scene_bootstrap(self):
        a=np.full((235,13,50,6),.1,np.float32)
        b=np.full((235,13,50,6),.3,np.float32)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"accuracy.npy";np.save(p,a)
            self.assertEqual(load_accuracy(p).shape,(235,13,50,6))
            np.save(p,np.ones((235,13,50,5),np.float32))
            with self.assertRaises(ValueError):
                load_accuracy(p)
        indices=np.random.default_rng(0).integers(0,235,size=(200,235))
        result=compare_arrays(a,b,indices)
        for row in result:
            self.assertAlmostEqual(row["delta_pp"],20,places=4)
            self.assertAlmostEqual(row["ci95_low_pp"],20,places=4)
            self.assertAlmostEqual(row["ci95_high_pp"],20,places=4)


class FakeGC6D:
    camera="realsense-d435"
    samples=[(7,2,0),(7,6,1)]
    scene_ids=[7]
    def __len__(self):return len(self.samples)


class TestCollisionDumpProtocol(unittest.TestCase):
    def test_identity_and_finite_nx17(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            paths=expected_paths(FakeGC6D(),root,"on",[0,1])
            for row in paths:
                atomic_npy(row[3],np.ones((2,17),np.float32))
                self.assertEqual(validate_grasps(row[3]).shape,(2,17))
            meta=full_identity_audit(paths,expected=2)
            self.assertEqual(meta["frames"],2)
            self.assertEqual(meta["total_saved_grasps"],4)
            self.assertEqual(meta["empty_frames"],0)
            paths[0][3].unlink()
            with self.assertRaises(FileNotFoundError):
                full_identity_audit(paths,expected=2)

    def test_manifest_requires_lease_and_refuses_different_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            v=dict(checkpoint_sha256="abc",filter_modes=["on"],official_eval_mu=[.2,.4,.6,.8,1,1.2])
            ensure_manifest(root,v,resume=False)
            ensure_manifest(root,v,resume=True)
            with self.assertRaises(FileExistsError):
                ensure_manifest(root,v,resume=False)
            with self.assertRaises(RuntimeError):
                ensure_manifest(root,dict(v,checkpoint_sha256="different"),resume=True)

    def test_explicit_preprocessing_acknowledgment(self):
        args=parse_args([
            "--historical-root","/nonexistent","--checkpoint","/nonexistent",
            "--gc6d-root","/nonexistent","--output-root","/tmp/test",
            "--collision","on","--collision-thresh",".01",
        ])
        self.assertFalse(args.accept_preprocessing_privilege)
        self.assertEqual(args.collision,"on")
        self.assertAlmostEqual(args.collision_thresh,.01)


class FakeDataset(Dataset):
    def __init__(self):
        self.scenename=["scene_1","scene_1","scene_2","scene_2"]
        self.frameid=[0,10,0,10]
    def __len__(self):return 4
    def __getitem__(self,idx):
        return float(np.random.random()),float(torch.rand(1).item())


class TestSourceSamplingAndGradients(unittest.TestCase):
    def test_subset_scene_sampling(self):
        base=FakeDataset()
        subset=Subset(base,[0,1,2,3])
        samples=sample_source_indices(subset,2,seed=0)
        self.assertEqual(len(samples),2)
        self.assertEqual(len({base.scenename[subset.indices[i]] for i in samples}),2)
        self.assertEqual(samples,sample_source_indices(subset,2,seed=0))

    def test_per_index_rng_is_stable_and_restored(self):
        base=FakeDataset()
        ds=DeterministicIndexedDataset(base,[0,1,2,3])
        first=ds[1];second=ds[1]
        self.assertEqual(first,second)

    def test_cosine_zero_norm_is_undefined_not_negative(self):
        x=torch.tensor([1.,0.]);y=torch.tensor([-1.,0.])
        dot,nx,ny,n=dot_grads([x],[y])
        self.assertAlmostEqual(dot/(nx*ny),-1.)
        self.assertEqual(n,1)
        dot,nx,ny,n=dot_grads([None],[y])
        self.assertEqual(nx,0.)
        self.assertEqual(n,0)


if __name__=="__main__":
    unittest.main()
