#!/usr/bin/env python3
"""P0-A: evaluator-quality labels for *saved* ranked GraspNet predictions.

Produces candidate_rows.csv for mixed_p0_a_rank_stats.py --candidate-csv.
This is a supplementary fixed-candidate probe: GraspNet CAD collision and
DexNet friction scoring are used, but the list-selection semantics (global
Top-K after GraspGroup NMS) are not claimed to equal the official AP evaluator
(which also uses object assignment and object-wise selection).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# Older graspnetAPI uses removed NumPy aliases.
if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]
from graspnetAPI import GraspGroup, GraspNetEval
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.eval_utils import (
    collision_detection, compute_closest_points, create_table_points,
    get_grasp_score, transform_points, voxel_sample_points,
)
from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory


SPLITS = dict(test_seen=100, test_similar=130, test_novel=160)
DOMAINS = ("graspnet", "gntrans")
FRICTIONS = (1.2, 1.0, .8, .6, .4, .2)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graspnet-root", required=True)
    p.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    p.add_argument("--dump-template", default="{run}/{domain}/test_collision_off",
                   help="Directory under which scene_XXXX/realsense/NNNN.npy dumps exist.")
    p.add_argument("--camera", default="realsense")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--max-scenes-per-split", type=int, default=3,
                   help="0 processes all 30 scenes; default deliberately small.")
    p.add_argument("--sample-every", type=int, default=50,
                   help="Use sampled annotations 0..250, divisibility checked.")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--eval-chunk", type=int, default=96)
    p.add_argument("--use-nms", type=int, choices=(0,1), default=1)
    return p.parse_args()


class QualityEvaluator:
    """Mirror the existing diagnose_cva_candidate_validity.RawCandidateEvaluator."""

    def __init__(self, root, camera, split, chunk):
        self.eval = GraspNetEval(root, camera, split=split)
        self.chunk = max(1, int(chunk))
        self.table = create_table_points(
            1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008
        )
        config = get_config()
        self.fc_configs = {}
        for mu in FRICTIONS:
            config["metrics"]["force_closure"]["friction_coef"] = mu
            self.fc_configs[round(mu,2)] = GraspQualityConfigFactory.create_config(
                config["metrics"]["force_closure"]
            )
        self.cache = {}

    def _models(self, sid):
        if sid not in self.cache:
            models, dexmodels, _ = self.eval.get_scene_models(sid, ann_id=0)
            self.cache[sid] = ([voxel_sample_points(x, .008) for x in models], dexmodels)
        return self.cache[sid]

    def score(self, scene_id, anno_id, grasps):
        n = len(grasps)
        if n == 0:
            return np.zeros(0, bool), np.zeros(0, bool), np.zeros(0, np.float32)
        models, dexmodels = self._models(scene_id)
        _, poses, camera_pose, align_mat = self.eval.get_model_poses(scene_id, anno_id)
        if len(models) != len(poses):
            raise ValueError(f"Scene pose/model mismatch: {scene_id}/{anno_id}")
        cam_models = [transform_points(m, poses[i]) for i, m in enumerate(models)]
        scene = np.concatenate(cam_models, axis=0)
        obj_idx = np.concatenate([
            np.full(len(x), i, np.int64) for i, x in enumerate(cam_models)
        ])
        assigned = obj_idx[compute_closest_points(grasps[:, 13:16], scene)]
        table_cam = transform_points(
            self.table, np.linalg.inv(align_mat @ camera_pose)
        )
        full_scene = np.concatenate((scene, table_cam), axis=0)
        coll = np.zeros(n, dtype=bool)
        empty = np.zeros(n, dtype=bool)
        fric = np.full(n, -1, np.float32)
        for obj in range(len(models)):
            idx = np.flatnonzero(assigned == obj)
            for start in range(0, len(idx), self.chunk):
                sel = idx[start:start+self.chunk]
                if len(sel) == 0:
                    continue
                c_l, e_l, d_l = collision_detection(
                    [grasps[sel]], [cam_models[obj]], [dexmodels[obj]],
                    [poses[obj]], full_scene, outlier=.05, return_dexgrasps=True,
                )
                c, e, dex = np.asarray(c_l[0],bool), np.asarray(e_l[0],bool), d_l[0]
                coll[sel], empty[sel] = c, e
                for j, out_idx in enumerate(sel):
                    if c[j] or dex[j] is None:
                        continue
                    fric[out_idx] = float(get_grasp_score(
                        dex[j], dexmodels[obj], np.asarray(FRICTIONS),
                        self.fc_configs
                    ))
        return coll, empty, fric


def index_dumps(root, camera):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    index = {}
    for file in root.rglob(f"scene_*/{camera}/*.npy"):
        sid = int(file.parent.parent.name.split("_")[-1])
        frame = int(file.stem)
        key = (sid, frame)
        if key in index:
            raise ValueError(f"Ambiguous prediction dump for {key}: {index[key]} and {file}")
        index[key] = file
    return index


def main():
    a = parse_args()
    if a.sample_every <= 0 or a.topk <= 0 or a.topk > 50 or a.max_scenes_per_split < 0:
        raise ValueError("Invalid --sample-every, --topk or --max-scenes-per-split.")
    # Evaluate only the same 10%-sampled annotation frames as the AP comparison.
    annos = tuple(range(0,256,a.sample_every))
    if any(x % 10 != 0 for x in annos):
        raise ValueError("Candidate frames must be a subset of stride-10 official AP frames")
    runs = {}
    for spec in a.run:
        if "=" not in spec:
            raise ValueError("--run requires NAME=DIR")
        name, root = spec.split("=",1)
        if name in runs or not name:
            raise ValueError(f"Repeated/empty run name: {name}")
        runs[name] = str(Path(root).resolve())
    if len(runs) != 2:
        raise ValueError("Pass two run roots.")

    out = Path(a.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run","domain","split","scene","frame","rank",
                  "original_score","friction","collision","empty","is_nms",
                  "candidate_selection"]
    summary = []
    with out.open("w", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        writer.writeheader()
        for domain in DOMAINS:
            for split, start in SPLITS.items():
                evaluator = QualityEvaluator(a.graspnet_root, a.camera, split, a.eval_chunk)
                count_scenes = 30 if not a.max_scenes_per_split else a.max_scenes_per_split
                for name, root in runs.items():
                    dumpdir = a.dump_template.format(run=root, domain=domain, split=split)
                    index = index_dumps(dumpdir, a.camera)
                    written = 0
                    for sid in range(start,start+count_scenes):
                        for anno in annos:
                            key = sid, anno
                            if key not in index:
                                raise FileNotFoundError(f"Missing prediction for {name}/{domain}/{split}: {key}, dump={dumpdir}")
                            grasps = np.load(index[key], allow_pickle=False)
                            if grasps.ndim != 2 or grasps.shape[1] != 17 or not np.isfinite(grasps).all():
                                raise ValueError(f"Bad Nx17 predictions at {index[key]}")
                            gg = GraspGroup(grasps)
                            if a.use_nms:
                                gg = gg.nms()
                            vals = gg.grasp_group_array
                            order = np.argsort(-vals[:,0], kind="stable")[:a.topk]
                            vals = vals[order]
                            col, empty, friction = evaluator.score(sid, anno, vals)
                            for i in range(len(vals)):
                                writer.writerow({
                                    "run":name,"domain":domain,"split":split,
                                    "scene":f"scene_{sid:04d}","frame":f"{anno:04d}",
                                    "rank":i+1, "original_score":float(vals[i,0]),
                                    "friction":float(friction[i]), "collision":int(col[i]),
                                    "empty":int(empty[i]), "is_nms":int(a.use_nms),
                                    "candidate_selection":"NMS+global-topK" if a.use_nms else "global-topK",
                                })
                                written += 1
                    summary.append(dict(run=name,domain=domain,split=split,rows=written))
                    print(f"[candidate probe] {name}/{domain}/{split}: {written} rows", flush=True)
    out.with_suffix(".json").write_text(json.dumps(dict(
        source="official GraspNet collision/DexNet quality adapted from repo diagnostic",
        type="nonofficial_fixed_candidate_oracle",
        caveat="NMS/global Top-K does NOT exactly match official object-wise AP selection",
        rank_mu=FRICTIONS, scene_count=30 if not a.max_scenes_per_split else a.max_scenes_per_split,
        annotations=annos, output=str(out), results=summary
    ), indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
