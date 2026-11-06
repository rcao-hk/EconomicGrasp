"""Synthetic CPU tests: native CDF lattice, query pairing, width/depth controls."""
from __future__ import annotations
import unittest
import numpy as np
import torch

from diagnostics.rayrope_action_lattice import (
    build_p2_actions, count_width_status, decode_cdf_lattice,
    physical_action_id, select_p2_queries,
)
from diagnostics.analyze_rayrope_p2_lattice import (
    query_winner_pairs, sweep_families, summarize_sweep,
    bootstrap_scene_delta, per_frame_summary,
)
from diagnostics.audit_rayrope_width import audit_widths
from moge_rayrope.attention import GraspRayRoPEGrouping
from moge_rayrope.config import ModelConfig


def identity_rotation(approach, angle):
    # Deterministic valid SO(3) frame: rotate in-plane around the Z axis.
    n=len(angle)
    rot=torch.eye(3, dtype=angle.dtype, device=angle.device)[None].expand(n,3,3).clone()
    c,s=torch.cos(angle),torch.sin(angle)
    rot[:,0,0]=c;rot[:,0,1]=-s
    rot[:,1,0]=s;rot[:,1,1]=c
    return rot


def fixture_lattice():
    Q, A, D = 4, 3, 2
    xyz = torch.tensor([[[0, 0, 0.5], [0.01, 0, 0.5],
                         [0.02, 0, 0.5], [0.03, 0, 0.5]]], dtype=torch.float)
    views = torch.tensor([[[0, 0, -1]]], dtype=torch.float).expand(1, Q, 3).clone()
    logits = torch.arange(Q*A*D, dtype=torch.float).reshape(1,1,Q,A,D).repeat(1,6,1,1,1) / 10
    widths = torch.full((1,D,Q,A),1.0)  # raw decoded width .12 -> clamps to .10
    return decode_cdf_lattice({
        "xyz_graspable":xyz, "grasp_top_view_xyz":views,
        "grasp_cdf_pred_angle_depth":logits,
        "grasp_width_pred_angle_depth":widths,
    }, 0, identity_rotation, max_width_m=0.1)


def parsed_row(row, condition, parent, query, quality):
    r = np.asarray(row)
    return {
        "sample":"scene0100_frame0000",
        "scene":100, "split":"seen",
        "condition":condition,
        "candidate":physical_action_id(r),
        "parent":parent, "score":float(r[0]), "quality":float(quality),
        "width_m":float(r[1]), "insertion":float(r[3]),
        "center":tuple(map(float,r[13:16])),
        "mu_flags":{m:bool(quality>0) for m in (0.2,0.4,0.6,0.8,1.0,1.2)},
        "failure":"success" if quality>0 else "empty_grasp",
        "raw":{"query_id":str(query),"parent_id":parent,
               "width_mode":"predicted" if condition=="query_winner" else "fixed"},
    }


class ActionLatticeTests(unittest.TestCase):
    def test_full_decode_geometry_and_width_clamp(self):
        x=fixture_lattice()
        self.assertEqual(x.grasp.shape,(24,17))
        self.assertEqual(int(x.query_winner.sum()),4)
        self.assertTrue(np.allclose(x.grasp[:,1],0.1))
        self.assertTrue(np.all(x.raw_width_m>0.1))
        self.assertTrue(np.allclose(x.grasp[:,3],0.01*(x.depth_id+1)))
        native=x.grasp[x.query_winner]
        self.assertEqual(len(native),4)
        same=native[0].copy();same[0]=0.0
        self.assertEqual(physical_action_id(native[0]),physical_action_id(same))

    def test_balanced_query_and_sweeps(self):
        x=fixture_lattice()
        a=select_p2_queries(x,num_query_anchors=3,nonwinner_per_query=1,seed=9)
        b=select_p2_queries(x,num_query_anchors=3,nonwinner_per_query=1,seed=9)
        self.assertEqual(a,b)
        self.assertEqual(len(a["query_winner"]),3)
        self.assertEqual(len(a["query_nonwinner"]),3)
        self.assertEqual(len(a["lattice_uniform"]),6)
        self.assertTrue(all(not x.query_winner[k] for k in a["query_nonwinner"]))
        records=build_p2_actions(x,a,widths_m=(0.04,0.06,0.08,0.10),max_width_m=0.1)
        width=[y for y in records if y.condition=="width_sweep"]
        insert=[y for y in records if y.condition=="insertion_sweep_fixed_width"]
        self.assertEqual(len(insert),3*x.num_depths)
        for q in a["anchor_query_ids"]:
            family=[y for y in width if y.query_id==q]
            self.assertEqual(len(family),4)  # predicted width .1 deduplicated
            self.assertEqual(len({round(float(y.row[0]),7) for y in family}),1)
            self.assertEqual(len({round(float(y.row[3]),7) for y in family}),1)
            ifam=[y for y in insert if y.query_id==q]
            self.assertEqual(len({round(float(y.row[1]),7) for y in ifam}),1)
            self.assertEqual(len({round(float(y.row[3]),7) for y in ifam}),x.num_depths)

    def test_cap_histogram_near_not_greater(self):
        widths=[0.08,0.1,0.100000002,0.1001]
        vals=count_width_status(widths,0.1,1e-6)
        self.assertEqual((vals["below_cap"],vals["near_cap"],vals["above_cap"]),(1,2,1))
        rows=[]
        for w in widths:
            rows.append({"condition":"native","width_m":w,"scene":100,"raw":{}})
        summary=audit_widths(rows,cap_m=0.1,tol_m=1e-6)
        self.assertEqual(summary["by_condition"]["native"]["above_cap"],1)
        self.assertEqual(summary["by_condition"]["native"]["at_or_above_nominal_bin_count"],3)

    def test_diagnostic_forward_mask_vs_coordinates(self):
        torch.manual_seed(0)
        cfg=ModelConfig(use_rayrope=True,ray_encoding="point",group_chunk=2,checkpoint_chunks=False)
        module=GraspRayRoPEGrouping(cfg,feature_dim=8,output_dim=16)
        module.eval()
        fmap=torch.randn(1,8,64,64)
        depth=torch.full((1,1,64,64),.5)
        module.geometry_context={"sigma":torch.zeros_like(depth)}
        kwargs=dict(
            seed_features=torch.randn(1,8,2),
            token_sel_idx=torch.tensor([[32*64+32,33*64+33]]),
            seed_xyz=torch.tensor([[[0.,0.,.5],[.005,0.,.5]]]),
            top_view_rot=torch.eye(3).reshape(1,1,3,3).expand(1,2,3,3).clone(),
            feat_map=fmap,depth_map=depth,
            camera_K=torch.tensor([[[160.,0.,31.5],[0.,160.,31.5],[0.,0.,1.]]]),
        )
        with torch.inference_mode():
            native=module(**kwargs)
            module.set_diagnostic("mask_only",.01)
            mask=module(**kwargs)
            mask_stats=module.last_diagnostic_stats
            self.assertEqual(mask_stats["mask_flip_count"],0)
            torch.testing.assert_close(mask,native,atol=0,rtol=0)
            module.set_diagnostic("coords_only",.01)
            coords=module(**kwargs)
            module.set_diagnostic("both",.01)
            both=module(**kwargs)
            torch.testing.assert_close(coords,both,rtol=0,atol=1e-7)
        self.assertTrue(torch.isfinite(coords).all())
        module.set_diagnostic(None)
        module.train()
        with self.assertRaises(RuntimeError):
            module.set_diagnostic("both",.01)

    def test_matched_query_and_bootstrap(self):
        x=fixture_lattice()
        a=select_p2_queries(x,num_query_anchors=2,nonwinner_per_query=1,seed=8)
        recs=build_p2_actions(x,a,widths_m=(.04,.06),max_width_m=.1)
        rows=[]
        for r in recs:
            qual=.8 if r.condition=="query_winner" else .2
            rows.append(parsed_row(r.row,r.condition,r.parent_id,r.query_id,qual))
        matched=query_winner_pairs(rows)
        self.assertEqual(len(matched),2)
        self.assertAlmostEqual(np.mean([z["quality_delta_winner_minus_nonwinner"] for z in matched]),.6)
        ws=sweep_families(rows,"width_sweep")
        self.assertEqual(len(ws),2)
        self.assertEqual(summarize_sweep(ws)["fraction_quality_changes"],0.0)
        # Synthetic second scene permits a proper 2-cluster interval.
        fr=[
            {"sample":"s1","scene":100,"split":"seen","condition":"query_winner","top1_mean_quality":.8},
            {"sample":"s1","scene":100,"split":"seen","condition":"query_nonwinner","top1_mean_quality":.2},
            {"sample":"s2","scene":101,"split":"seen","condition":"query_winner","top1_mean_quality":.9},
            {"sample":"s2","scene":101,"split":"seen","condition":"query_nonwinner","top1_mean_quality":.1},
        ]
        summary=bootstrap_scene_delta(fr,"query_winner","query_nonwinner","top1_mean_quality",n_boot=100,seed=1)
        self.assertEqual(summary["paired_scenes"],2)
        self.assertGreater(summary["delta"],0)
        self.assertIsNotNone(summary["ci95"])


if __name__=="__main__":
    unittest.main()
