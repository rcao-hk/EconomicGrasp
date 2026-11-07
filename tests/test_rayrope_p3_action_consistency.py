"""CPU-only synthetic tests for P3 action-consistent insertion diagnostics."""
from __future__ import annotations

import copy
import unittest
from math import isclose

import numpy as np

from diagnostics.analyze_rayrope_p3_action_consistency import (
    FIXED, PRED, WIDTH, NATIVE, analyze, _width_protocol_compare,
    scene_cluster_bootstrap,
)
from diagnostics.replay_rayrope_p3_predicted_width import construct_replay
from diagnostics.rayrope_action_lattice import physical_action_id


MUS=(0.2,0.4,0.6,0.8,1.0,1.2)
I=np.eye(3,dtype=np.float32)


def physical_row(center,depth,width,score):
    x,y,z=center
    return np.asarray([score,width,.02,depth,*I.reshape(-1),x,y,z,-1],dtype=np.float32)


def parsed_row(sample, scene, parent, mode, d, score, quality, width,
               selected_depth=1, center=(.01,0,.5)):
    g=physical_row(center,.01*(d+1),width,score)
    raw={
        "sample_id":sample, "scene_id":str(scene), "split":"test_seen",
        "candidate_id":physical_action_id(g),
        "condition":mode, "intervention":mode,
        "query_id":"3","angle_id":"7","depth_id":str(d),
        "parent_id":parent,"query_winner":str(int(d==selected_depth)),
        "score":str(float(g[0])),"width_m":str(float(g[1])),
        "insertion_depth_m":str(float(g[3])),
        "height_m":str(float(g[2])),
        "width_predicted_decoded_m":str([.04,.08,.06,.10][d]),
        "center_x":str(float(g[13])),"center_y":str(float(g[14])),"center_z":str(float(g[15])),
    }
    raw.update({f"r{i}{j}":str(float(I[i,j])) for i in range(3) for j in range(3)})
    flags={mu:(float(quality)*6 >= (6-i)-1e-6) for i,mu in enumerate(MUS)}
    failure="success" if quality>0 else "empty_grasp"
    return {
        "raw":raw,"sample":sample,"scene":scene,"split":"seen",
        "candidate":raw["candidate_id"],"condition":mode,
        "parent":parent,"score":float(g[0]),"quality":float(quality),
        "friction":0.6 if quality>0 else -1.0,"mu_flags":flags,
        "failure":failure,"width_m":float(g[1]),"insertion":float(g[3]),
        "center":tuple(float(v) for v in g[13:16]),
        "u":float("nan"),"v":float("nan"),
    }


def fixture():
    rows=[]
    scores=[.3,.9,.7,.6]
    fixed_quality=[0.0,2/6,1.0,1/6]
    pred_quality=[1/6,2/6,5/6,4/6]
    widths=[.04,.08,.06,.10]
    for scene,sample in [(100,"scene0100_frame0000"),(101,"scene0101_frame0128")]:
        parent=f"q0003-a07-w0.080000"
        for d,score in enumerate(scores):
            rows.append(parsed_row(sample,scene,parent,FIXED,d,score,fixed_quality[d],.08))
            rows.append(parsed_row(sample,scene,parent,PRED,d,score,pred_quality[d],widths[d]))
        # Distinct query winner entries in each frame, with rank-2 better than rank-1.
        for idx,(score,quality) in enumerate(((.95,2/6),(.85,5/6))):
            row=parsed_row(sample,scene,f"winner_{idx}",NATIVE,idx,score,quality,.08)
            row["raw"]["query_id"]=str(idx+7)
            row["candidate"]=physical_action_id(physical_row((.01*(idx+1),0,.5),.01*(idx+1),.08,score))
            row["raw"]["candidate_id"]=row["candidate"]
            rows.append(row)
        # Width sweep at a fixed physical action; predicted width is .08.
        for w,q,mode in ((.04,1/6,"fixed"),(.06,2/6,"fixed"),
                         (.08,3/6,"predicted"),(.10,1/6,"fixed")):
            row=parsed_row(sample,scene,f"width_q3_a7_d1",WIDTH,1,.9,q,w)
            row["raw"]["width_mode"]=mode
            rows.append(row)
    return rows


class P3DiagnosticsTests(unittest.TestCase):
    def test_same_CDF_insertion_protocols_match_with_width_only(self):
        rows=fixture()
        result=analyze(rows,n_boot=100,seed=7)
        mode=result["fixed_vs_predicted_width"]
        self.assertTrue(mode["available"])
        self.assertEqual(mode["families"],2)
        self.assertAlmostEqual(mode["mean_fixed_regret"],1.0-2/6)
        self.assertAlmostEqual(mode["mean_predicted_regret"],5/6-2/6)
        self.assertAlmostEqual(mode["pred_minus_fixed_regret"]["mean"],-1/6)
        self.assertAlmostEqual(mode["pred_minus_fixed_oracle"]["mean"],-1/6)
        self.assertTrue(mode["pred_minus_fixed_regret"]["ci95"] is not None)
        self.assertEqual(mode["fixed"]["failure_attribution"]["n_positive_regret"],2)
        self.assertEqual(mode["fixed"]["depth_distribution"]["by_depth"]["depth_3cm"]["num_actions"],2)

    def test_refuse_geometry_or_score_or_selected_width_mismatch(self):
        for attribute,value in (("score",.1),("center",(.2,.1,.5)),("width_m",.075)):
            rows=fixture()
            target=next(x for x in rows if x["condition"]==PRED and x["sample"]=="scene0100_frame0000" and x["raw"]["depth_id"]=="1")
            target[attribute]=value
            if attribute=="width_m":
                # Same physical action ID must agree at the selected bin.
                target["candidate"]=physical_action_id(physical_row((.01,0,.5),.02,.075,.9))
            with self.assertRaises(ValueError):
                analyze(rows,n_boot=20)

    def test_rank1_vs_rank2_and_width_baselines(self):
        data=analyze(fixture(),n_boot=100,seed=3)
        rank=data["rank1_vs_rank2"]
        self.assertTrue(rank["available"])
        self.assertEqual(rank["frames"],2)
        self.assertAlmostEqual(rank["rank2_minus_rank1"]["mean"],3/6)
        self.assertEqual(rank["rank2_better_fraction"],1.0)
        width=data["width_predicted_vs_constants"]
        self.assertTrue(width["available"])
        self.assertEqual(width["families"],2)
        self.assertAlmostEqual(width["pred_regret"]["mean"],0.0)
        self.assertAlmostEqual(width["constant_baselines"]["0.10"]["pred_minus_constant_quality"]["mean"],2/6)

    def test_fallback_when_predicted_sweep_absent(self):
        rows=[x for x in fixture() if x["condition"]!=PRED]
        result=analyze(rows,n_boot=30,seed=1)
        self.assertFalse(result["fixed_vs_predicted_width"]["available"])
        self.assertIsNotNone(result["fixed_only_when_unpaired"])
        self.assertTrue(result["rank1_vs_rank2"]["available"])

    def test_replay_constructs_predicted_width_and_protects_native_action(self):
        rows=[x for x in fixture() if x["condition"]==FIXED]
        pairs=construct_replay(rows)
        self.assertEqual(len(pairs),8)
        for source,new_row in pairs:
            d=int(source["raw"]["depth_id"])
            self.assertAlmostEqual(float(new_row[1]),[.04,.08,.06,.10][d],places=6)
            self.assertAlmostEqual(float(new_row[0]),source["score"],places=6)
            self.assertAlmostEqual(float(new_row[3]),source["insertion"],places=6)
        broken=copy.deepcopy(rows)
        broken[0]["raw"].pop("width_predicted_decoded_m")
        with self.assertRaises(KeyError):
            construct_replay(broken)
        broken=copy.deepcopy(rows)
        selected=next(x for x in broken if x["raw"]["depth_id"]=="1")
        selected["raw"]["width_predicted_decoded_m"]="0.09"
        with self.assertRaises(ValueError):
            construct_replay(broken)

    def test_scene_cluster_bootstrap_clusters_frames(self):
        records=[]
        for scene in (100,101):
            for frame in (0,1):
                records.append({"scene":scene,"split":"seen","sample":f"{scene}_{frame}",
                                "delta":float(scene-99)})
        result=scene_cluster_bootstrap(records,"delta",n_boot=100,seed=9)
        self.assertEqual(result["num_scenes"],2)
        self.assertEqual(result["num_frames"],4)
        self.assertAlmostEqual(result["mean"],1.5)


if __name__=="__main__":
    unittest.main()
