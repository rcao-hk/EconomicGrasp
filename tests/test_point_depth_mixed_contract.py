"""Dependency-light tests for the controlled point-depth mixed baseline.

Run on CPU: pytest -q tests/test_point_depth_mixed_contract.py
"""
from __future__ import annotations

import numpy as np
import pytest

from utils.point_depth_mixed_contract import (
    assert_paired_frames,
    network_rgb_pose_inputs,
    scene_frame_signature,
    scene_stride_indices,
)
from tools.summarize_point_depth_pose_ap import metrics, paired_scene_bootstrap


class FakeDataset:
    def __init__(self, scene_start, scenes=100):
        self.scenename=[]
        self.frameid=[]
        for s in range(scene_start,scene_start+scenes):
            self.scenename.extend([f"scene_{s:04d}"]*256)
            self.frameid.extend(range(256))
    def __len__(self):
        return len(self.frameid)


def test_mixed_pairing_10pct_and_heldout_seen():
    real=FakeDataset(0)
    trans=FakeDataset(0)
    seen=FakeDataset(100,30)
    ia=scene_stride_indices(real,0.1)
    ib=scene_stride_indices(trans,0.1)
    iv=scene_stride_indices(seen,0.1)
    assert len(ia)==len(ib)==2600 and len(iv)==780
    assert [real.frameid[i] for i in ia[:26]]==list(range(0,256,10))
    assert set(real.scenename).isdisjoint(seen.scenename)
    assert assert_paired_frames(real,trans,ia,ib,"train")==2600
    assert scene_frame_signature(real,ia)==scene_frame_signature(trans,ib)
    assert len(scene_stride_indices(real,1.0))==25600
    assert scene_stride_indices(real,0.5)[:4]==[0,2,4,6]


def test_bad_domain_pairing_fails():
    a,b=FakeDataset(0,1),FakeDataset(0,1)
    b.frameid[10]=256  # unique, but removes the selected frame 10
    with pytest.raises(RuntimeError,match="frame IDs differ"):
        assert_paired_frames(a,b,scene_stride_indices(a,0.1),
                             scene_stride_indices(b,0.1),"train")
    with pytest.raises(ValueError):
        scene_stride_indices(a,0)
    with pytest.raises(ValueError):
        scene_stride_indices(a,1.1)


def test_network_input_allowlist_excludes_sensor_and_gt():
    batch={
        "img":"rgb","K":"calibration","img_idxs":"pixel_indices",
        "camera_pose_vec":"global_cam_pose", "camera_gravity_vec":"world_up",
        "point_clouds":"sensor_cloud",
        "sensor_depth_m":"sensor_depth",
        "gt_depth_m":"rendered_gt",
        "depth_prob_gt":"supervision_only",
        "scene_idx":101,
    }
    keep=network_rgb_pose_inputs(batch,pose_depth_mode="global_film")
    assert keep=={
        "img":"rgb","K":"calibration","img_idxs":"pixel_indices",
        "camera_pose_vec":"global_cam_pose","scene_idx":101,
    }
    with pytest.raises(KeyError,match="camera_pose_vec"):
        network_rgb_pose_inputs({"img":1,"K":2,"img_idxs":3},pose_depth_mode="global_film")
    with pytest.raises(ValueError):
        network_rgb_pose_inputs(batch,pose_depth_mode="bogus")


def test_ap_shape_aggregation_and_paired_scene_ci():
    a=np.ones((30,26,50,6),dtype=float)
    b=np.zeros_like(a)
    assert metrics(a)["mean_ap_pct_mu_0p2_to_1p0"]==100.0
    assert metrics(a)["ap_mu_0p4_pct"]==100.0
    assert metrics(a)["prefix_1_mean_ap_pct"]==100.0
    names=("test_seen","test_similar","test_novel")
    summary=paired_scene_bootstrap({k:a for k in names},
                                   {k:b for k in names},
                                   replicates=128,seed=3)
    assert summary["mean_delta_ap_pp"]==100.0
    assert summary["mean_ci95_pp"]==[100.0,100.0]
